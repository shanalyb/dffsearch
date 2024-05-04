from clickhouse_connect import get_client
from langchain.schema import AIMessage, HumanMessage

class ClickHouseChatMessageHistory:
    def __init__(self, host, port, table_name, session_id):
        self.client = get_client(host=host, port=port)
        self.table_name = table_name
        self.session_id = session_id
        self._create_table_if_not_exists()

    def _create_table_if_not_exists(self):
        query = f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id UUID,
                session_id String,
                sender String,
                intent String,
                node_name String,
                message String,
                ins_timestamp DateTime
            ) ENGINE = MergeTree()
            PARTITION BY toYYYYMM(ins_timestamp)
            ORDER BY (session_id, id)
        """
        self.client.command(query)

    def add_message(self, message, intent, node_name, sender):
        query = f"""
            INSERT INTO {self.table_name} (id, session_id, sender, intent, node_name, message, ins_timestamp)
            VALUES (generateUUIDv4(), '{self.session_id}', '{sender}', '{intent}', '{node_name}', '{message.content}', now())
        """
        self.client.command(query)

    def add_user_message(self, message, intent, node_name):
        self.add_message(HumanMessage(content=message), intent, node_name, sender='Human')

    def add_ai_message(self, message, intent, node_name):
        self.add_message(AIMessage(content=message), intent, node_name, sender='Ai')
