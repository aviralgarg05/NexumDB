use super::types::Statement;

#[derive(Debug, Clone)]
pub enum Plan {
    CreateTable {
        name: String,
        columns: Vec<(String, String)>,
    },
    Insert {
        table: String,
        rows: usize,
    },
    Select {
        table: String,
        columns: Vec<String>,
    },
    ShowTables,
    DescribeTable {
        name: String,
    },
    DropTable {
        name: String,
        if_exists: bool,
    },
    Delete {
        table: String,
        has_where: bool,
    },
    Update {
        table: String,
        columns: Vec<String>,
        has_where: bool,
    },
}

pub struct Planner;

impl Planner {
    pub fn plan(statement: Statement) -> Plan {
        match statement {
            Statement::CreateTable { name, columns } => {
                let cols = columns
                    .iter()
                    .map(|c| (c.name.clone(), format!("{:?}", c.data_type)))
                    .collect();
                Plan::CreateTable {
                    name,
                    columns: cols,
                }
            }
            Statement::Insert { table, values, .. } => Plan::Insert {
                table,
                rows: values.len(),
            },
            Statement::Select {
                table, projection, ..
            } => {
                let columns = projection
                    .iter()
                    .map(|item| match item {
                        crate::sql::types::SelectItem::Wildcard => "*".to_string(),
                        crate::sql::types::SelectItem::Column { name, alias } => alias
                            .as_ref()
                            .map(|a| format!("{} AS {}", name, a))
                            .unwrap_or_else(|| name.clone()),
                    })
                    .collect();
                Plan::Select { table, columns }
            }
            Statement::ShowTables => Plan::ShowTables,
            Statement::DescribeTable { name } => Plan::DescribeTable { name },
            Statement::DropTable { name, if_exists } => Plan::DropTable { name, if_exists },
            Statement::Delete {
                table,
                where_clause,
            } => Plan::Delete {
                table,
                has_where: where_clause.is_some(),
            },
            Statement::Update {
                table,
                assignments,
                where_clause,
            } => Plan::Update {
                table,
                columns: assignments.iter().map(|(col, _)| col.clone()).collect(),
                has_where: where_clause.is_some(),
            },
        }
    }
}
