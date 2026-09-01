package secondary

import (
	"context"
	"fmt"
	"math/rand/v2"
	"sort"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus/pkg/v2/common"
	"github.com/milvus-io/milvus/pkg/v2/proto/indexpb"
	"github.com/milvus-io/milvus/pkg/v2/streaming/util/message"
	"github.com/samber/lo"
	"go.uber.org/zap"

	"github.com/zilliztech/milvus-backup/core/proto/backuppb"
	"github.com/zilliztech/milvus-backup/core/restore/conv"
	"github.com/zilliztech/milvus-backup/internal/client/milvus"
	"github.com/zilliztech/milvus-backup/internal/log"
	"github.com/zilliztech/milvus-backup/internal/namespace"
	"github.com/zilliztech/milvus-backup/internal/pbconv"
)

type collDDLTask struct {
	taskID string

	backupInfo *backuppb.BackupInfo
	dbBackup   *backuppb.DatabaseBackupInfo
	collBackup *backuppb.CollectionBackupInfo

	streamCli milvus.Stream
	logger    *zap.Logger
}

type ddlTaskArgs struct {
	TaskID string

	BackupInfo *backuppb.BackupInfo

	StreamCli milvus.Stream
}

func newCollDDLTask(args ddlTaskArgs, dbBackup *backuppb.DatabaseBackupInfo, collBackup *backuppb.CollectionBackupInfo) *collDDLTask {
	ns := namespace.New(collBackup.GetDbName(), collBackup.GetCollectionName())

	return &collDDLTask{
		taskID: args.TaskID,

		backupInfo: args.BackupInfo,
		dbBackup:   dbBackup,
		collBackup: collBackup,

		streamCli: args.StreamCli,
		logger:    log.With(zap.String("task_id", args.TaskID), zap.String("ns", ns.String())),
	}
}

func (ddlt *collDDLTask) Execute(ctx context.Context) error {
	if err := ddlt.createColl(ctx); err != nil {
		return fmt.Errorf("collection: create collection: %w", err)
	}

	if err := ddlt.createIndexes(ctx); err != nil {
		return fmt.Errorf("collection: create indexes: %w", err)
	}

	return nil
}

func (ddlt *collDDLTask) createIndexes(ctx context.Context) error {
	for _, index := range ddlt.collBackup.GetIndexInfos() {
		if err := ddlt.createIndex(ctx, index); err != nil {
			return fmt.Errorf("collection: create index: %w", err)
		}
	}

	return nil
}

// resolveIndexField returns the field the index belongs to. Backups taken
// without --backup_index_extra carry only the field name (field_id, type
// params and index params come from etcd and are absent), so fall back to the
// collection schema instead of broadcasting FieldID 0, which would attach the
// index to a nonexistent field and leave the import job stuck in IndexBuilding.
func (ddlt *collDDLTask) resolveIndexField(index *backuppb.IndexInfo) (*backuppb.FieldSchema, error) {
	schema := ddlt.collBackup.GetSchema()

	if id := index.GetFieldId(); id != 0 {
		for _, field := range schema.GetFields() {
			if field.GetFieldID() == id {
				return field, nil
			}
		}
		// A struct array's sub-fields are indexed in their own right and carry
		// their own field ids, but they live under the struct rather than in
		// the flat field list.
		for _, structField := range schema.GetStructArrayFields() {
			for _, field := range structField.GetFields() {
				if field.GetFieldID() == id {
					return field, nil
				}
			}
		}
		return nil, unknownIndexFieldErr(index)
	}

	name := index.GetFieldName()
	for _, field := range schema.GetFields() {
		if field.GetName() == name {
			return field, nil
		}
	}
	// DescribeIndex names a sub-field as struct[sub], while the schema holds it
	// under its bare name, so match both.
	for _, structField := range schema.GetStructArrayFields() {
		for _, field := range structField.GetFields() {
			if field.GetName() == name ||
				fmt.Sprintf("%s[%s]", structField.GetName(), field.GetName()) == name {
				return field, nil
			}
		}
	}
	return nil, unknownIndexFieldErr(index)
}

func unknownIndexFieldErr(index *backuppb.IndexInfo) error {
	return fmt.Errorf("collection: index %s refers to unknown field (id=%d name=%s)",
		index.GetIndexName(), index.GetFieldId(), index.GetFieldName())
}

// indexParamsOf returns the index params to broadcast. Prefer the etcd copy
// (IndexParams); otherwise rebuild them from the DescribeIndex params map,
// which already carries index_type / metric_type and the index-specific keys.
func indexParamsOf(index *backuppb.IndexInfo) []*commonpb.KeyValuePair {
	if len(index.GetIndexParams()) != 0 {
		return pbconv.BakKVToMilvusKV(index.GetIndexParams())
	}
	keys := lo.Keys(index.GetParams())
	sort.Strings(keys)
	kvs := make([]*commonpb.KeyValuePair, 0, len(keys))
	for _, k := range keys {
		kvs = append(kvs, &commonpb.KeyValuePair{Key: k, Value: index.GetParams()[k]})
	}
	return kvs
}

func (ddlt *collDDLTask) createIndex(ctx context.Context, index *backuppb.IndexInfo) error {
	field, err := ddlt.resolveIndexField(index)
	if err != nil {
		return err
	}
	typeParams := pbconv.BakKVToMilvusKV(index.GetTypeParams())
	if len(typeParams) == 0 {
		typeParams = pbconv.BakKVToMilvusKV(field.GetTypeParams())
	}
	if index.GetFieldId() == 0 {
		ddlt.logger.Info("index info has no field id (backup taken without --backup_index_extra), resolved from schema",
			zap.String("index_name", index.GetIndexName()), zap.String("field_name", field.GetName()),
			zap.Int64("field_id", field.GetFieldID()))
	}

	indexParams := indexParamsOf(index)
	userIndexParams := pbconv.BakKVToMilvusKV(index.GetUserIndexParams())
	if len(userIndexParams) == 0 {
		// DescribeIndex reports the user-facing params from user_index_params;
		// without them the target shows the index as AUTOINDEX.
		userIndexParams = indexParams
	}

	indexInfo := &indexpb.IndexInfo{
		CollectionID:    ddlt.collBackup.GetCollectionId(),
		FieldID:         field.GetFieldID(),
		IndexName:       index.GetIndexName(),
		IndexID:         index.GetIndexId(),
		TypeParams:      typeParams,
		IndexParams:     indexParams,
		IsAutoIndex:     index.GetIsAutoIndex(),
		UserIndexParams: userIndexParams,
		MinIndexVersion: index.GetMinIndexVersion(),
		MaxIndexVersion: index.GetMaxIndexVersion(),
	}
	fieldIndex := &indexpb.FieldIndex{IndexInfo: indexInfo, CreateTime: index.GetCreateTime()}
	body := &message.CreateIndexMessageBody{FieldIndex: fieldIndex}
	header := &message.CreateIndexMessageHeader{
		DbId:         ddlt.dbBackup.GetDbId(),
		CollectionId: ddlt.collBackup.GetCollectionId(),
		FieldId:      field.GetFieldID(),
		IndexId:      index.GetIndexId(),
		IndexName:    index.GetIndexName(),
	}

	ddlt.logger.Info("create index", zap.Any("header", header), zap.Any("body", body))

	builder := message.NewCreateIndexMessageBuilderV2().
		WithHeader(header).
		WithBody(body).
		WithBroadcast([]string{ddlt.backupInfo.GetControlChannelName()})

	err = ddlt.streamCli.Send(ctx, func(uint64) []message.MutableMessage {
		broadcast := builder.MustBuildBroadcast().WithBroadcastID(rand.Uint64())
		return broadcast.SplitIntoMutableMessage()
	})
	if err != nil {
		return fmt.Errorf("collection: broadcast create index: %w", err)
	}

	return nil
}

func (ddlt *collDDLTask) partitionNames() []string {
	return lo.Map(ddlt.collBackup.GetPartitionBackups(), func(part *backuppb.PartitionBackupInfo, _ int) string {
		return part.GetPartitionName()
	})
}

func (ddlt *collDDLTask) partitionIDs() []int64 {
	return lo.Map(ddlt.collBackup.GetPartitionBackups(), func(part *backuppb.PartitionBackupInfo, _ int) int64 {
		return part.GetPartitionId()
	})
}

func (ddlt *collDDLTask) createColl(ctx context.Context) error {
	header := &message.CreateCollectionMessageHeader{
		CollectionId: ddlt.collBackup.GetCollectionId(),
		DbId:         ddlt.dbBackup.GetDbId(),
		PartitionIds: ddlt.partitionIDs(),
	}

	ddlt.logger.Info("create collection", zap.Any("header", header))

	schema, err := conv.Schema(ddlt.collBackup.GetSchema())
	if err != nil {
		return fmt.Errorf("secondary: convert schema: %w", err)
	}
	if err := checkDynamicField(schema); err != nil {
		return err
	}
	schema.Properties = append(schema.Properties, &commonpb.KeyValuePair{
		Key:   common.ConsistencyLevel,
		Value: ddlt.collBackup.GetConsistencyLevel().String(),
	})
	appendSysFields(schema)

	ddlt.logger.Info("collection schema", zap.Any("schema", schema))

	err = ddlt.streamCli.Send(ctx, func(ts uint64) []message.MutableMessage {
		req := &message.CreateCollectionRequest{
			Base: &commonpb.MsgBase{
				MsgType:   commonpb.MsgType_CreateCollection,
				Timestamp: ts,
			},
			DbName:               ddlt.collBackup.GetDbName(),
			CollectionName:       ddlt.collBackup.GetCollectionName(),
			DbID:                 ddlt.dbBackup.GetDbId(),
			CollectionID:         ddlt.collBackup.GetCollectionId(),
			VirtualChannelNames:  ddlt.collBackup.GetVirtualChannelNames(),
			PhysicalChannelNames: ddlt.collBackup.GetPhysicalChannelNames(),
			CollectionSchema:     schema,
			PartitionNames:       ddlt.partitionNames(),
			PartitionIDs:         ddlt.partitionIDs(),
		}
		ddlt.logger.Info("create collection", zap.Any("request", req))

		builder := message.NewCreateCollectionMessageBuilderV1().
			WithHeader(header).
			WithBody(req).
			WithBroadcast(append(ddlt.collBackup.GetVirtualChannelNames(), ddlt.backupInfo.GetControlChannelName()))

		broadcast := builder.MustBuildBroadcast().WithBroadcastID(rand.Uint64())
		return broadcast.SplitIntoMutableMessage()
	})
	if err != nil {
		return fmt.Errorf("secondary: broadcast create collection: %w", err)
	}

	return nil
}
