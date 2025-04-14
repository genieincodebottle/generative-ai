# Project Title
Tagging Text

## Description

Utilize the open-source large language model (LLM) provided by Ollama for text tagging, coupled with PostgreSQL for efficiently storing both user inputs and system responses.

## Installation

1. Install Ollama with the below given link

[Ollama](https://ollama.com/download)

2. Install PostgreSql with the below given link

[PostgreSql](https://www.postgresql.org/download/)

[PostgreSql_Installation_Instruction](https://www.postgresqltutorial.com/postgresql-getting-started/install-postgresql/
)

3. Script to create tables and input records at postgreSql. Create these tables in your schema available at the Postgresql using pgAdmin or Sql Shell.

[Create_table_tutorials](https://www.commandprompt.com/education/different-methods-to-create-a-table-in-postgresql/)

```sql
CREATE TABLE customer_calls (
    id SERIAL PRIMARY KEY,
    customer_id INT NOT NULL,
    call_details TEXT NOT NULL,
    call_time TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Assuming the table 'customer_calls' has been created
-- Format: (customer_id, call_details)

INSERT INTO customer_calls (customer_id, call_details) VALUES
(101, 'Customer expressed happiness about the quick resolution of their billing issue.'),
(102, 'Customer was delighted with the advanced features of our new product.'),
(103, 'Customer was upset about the delayed shipment and requested an expedited delivery.'),
(104, 'Customer felt neutral and inquired about the specifications of various models.'),
(105, 'Customer expressed dissatisfaction with the recent service, noting multiple unresolved issues.'),
(106, 'Customer joyfully reported that our product exceeded their expectations.'),
(107, 'Customer was frustrated due to a misunderstanding about warranty coverage.'),
(108, 'Customer was indifferent when discussing the upcoming software update details.'),
(109, 'Customer was angry about receiving the wrong order and demanded a prompt resolution.'),
(110, 'Customer was thrilled to hear about our loyalty program upgrades.');


CREATE TABLE customer_tagging
(
    user_id SERIAL PRIMARY KEY,
    call_details text NOT NULL,
    sentiment text NOT NULL,
    aggresiveness integer NOT NULL
)
```