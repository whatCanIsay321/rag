from pymilvus import MilvusClient
from pymilvus import MilvusClient, DataType
client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)
client.use_database(
    db_name="my_database_2"
)

schema = MilvusClient.create_schema(
    auto_id=False,
)
schema.add_field(field_name="my_id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="my_vector", datatype=DataType.FLOAT_VECTOR, dim=5)
schema.add_field(field_name="my_varchar", datatype=DataType.VARCHAR, max_length=512)
index_params = client.prepare_index_params()

index_params.add_index(
    field_name="my_id",
    index_type="AUTOINDEX"
)

index_params.add_index(
    field_name="my_vector",
    index_type="AUTOINDEX",
    metric_type="COSINE"
)
client.create_collection(
    collection_name="customized_setup_1",
    schema=schema,
    index_params=index_params,
    enable_dynamic_field=False

)

client.create_collection(
    collection_name="my_collection",
    dimension=5,
    enable_dynamic_field=False
)
