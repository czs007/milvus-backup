// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package secondary

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/zilliztech/milvus-backup/core/proto/backuppb"
)

func schemaWithStruct() *backuppb.CollectionSchema {
	return &backuppb.CollectionSchema{
		Fields: []*backuppb.FieldSchema{
			{FieldID: 100, Name: "id"},
			{FieldID: 101, Name: "title_vector"},
		},
		StructArrayFields: []*backuppb.StructArrayFieldSchema{
			{
				FieldID: 138,
				Name:    "duplicate_radar_struct",
				Fields: []*backuppb.FieldSchema{
					{FieldID: 139, Name: "chunk_number"},
					{FieldID: 141, Name: "chunk_vector"},
				},
			},
			{
				FieldID: 142,
				Name:    "title_struct",
				Fields: []*backuppb.FieldSchema{
					{FieldID: 145, Name: "chunk_vector"},
				},
			},
		},
	}
}

func taskWithSchema(schema *backuppb.CollectionSchema) *collDDLTask {
	return &collDDLTask{collBackup: &backuppb.CollectionBackupInfo{Schema: schema}}
}

func TestResolveIndexField(t *testing.T) {
	ddlt := taskWithSchema(schemaWithStruct())

	t.Run("a flat field resolves by id", func(t *testing.T) {
		field, err := ddlt.resolveIndexField(&backuppb.IndexInfo{FieldId: 101})
		assert.NoError(t, err)
		assert.Equal(t, "title_vector", field.GetName())
	})

	// A struct array's sub-fields are indexed in their own right but do not
	// appear in the flat field list, so a lookup that only walks that list
	// rejects an index the backup carries correctly.
	t.Run("a struct array sub-field resolves by id", func(t *testing.T) {
		field, err := ddlt.resolveIndexField(&backuppb.IndexInfo{FieldId: 141})
		assert.NoError(t, err)
		assert.EqualValues(t, 141, field.GetFieldID())
		assert.Equal(t, "chunk_vector", field.GetName())
	})

	t.Run("a sub-field of a later struct resolves by id", func(t *testing.T) {
		field, err := ddlt.resolveIndexField(&backuppb.IndexInfo{FieldId: 145})
		assert.NoError(t, err)
		assert.EqualValues(t, 145, field.GetFieldID())
	})

	// Without --backup_index_extra there is no field id, only the name
	// DescribeIndex reports, which for a sub-field is struct[sub].
	t.Run("a sub-field resolves by its qualified name", func(t *testing.T) {
		field, err := ddlt.resolveIndexField(&backuppb.IndexInfo{
			FieldName: "duplicate_radar_struct[chunk_vector]",
		})
		assert.NoError(t, err)
		assert.EqualValues(t, 141, field.GetFieldID())
	})

	t.Run("a flat field resolves by name", func(t *testing.T) {
		field, err := ddlt.resolveIndexField(&backuppb.IndexInfo{FieldName: "title_vector"})
		assert.NoError(t, err)
		assert.EqualValues(t, 101, field.GetFieldID())
	})

	t.Run("a field that is in neither list is reported", func(t *testing.T) {
		_, err := ddlt.resolveIndexField(&backuppb.IndexInfo{
			IndexName: "idx", FieldId: 999, FieldName: "nope",
		})
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "refers to unknown field")
		assert.Contains(t, err.Error(), "999")
	})
}
