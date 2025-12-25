use super::types::{Column, DataType, Statement, Value};
use anyhow::{anyhow, Result};
use sqlparser::ast::{self, ColumnDef, DataType as SqlDataType, Expr, Statement as SqlStatement};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser as SqlParser;

pub struct Parser;

impl Parser {
    pub fn parse(sql: &str) -> Result<Statement> {
        let dialect = GenericDialect {};
        let statements = SqlParser::parse_sql(&dialect, sql)?;

        if statements.is_empty() {
            return Err(anyhow!("No statements found"));
        }

        let stmt = &statements[0];
        Self::convert_statement(stmt)
    }

    fn convert_statement(stmt: &SqlStatement) -> Result<Statement> {
        match stmt {
            SqlStatement::CreateTable { name, columns, .. } => {
                let table_name = name.to_string();
                let cols = columns
                    .iter()
                    .map(Self::convert_column)
                    .collect::<Result<Vec<_>>>()?;
                Ok(Statement::CreateTable {
                    name: table_name,
                    columns: cols,
                })
            }
            SqlStatement::Insert {
                table_name,
                columns,
                source,
                ..
            } => {
                let table = table_name.to_string();
                let col_names = columns.iter().map(|c| c.to_string()).collect();

                let values = if let ast::SetExpr::Values(values) = &*source.body {
                    values
                        .rows
                        .iter()
                        .map(|row| {
                            row.iter()
                                .map(Self::convert_expr)
                                .collect::<Result<Vec<_>>>()
                        })
                        .collect::<Result<Vec<_>>>()?
                } else {
                    return Err(anyhow!("Unsupported INSERT format"));
                };

                Ok(Statement::Insert {
                    table,
                    columns: col_names,
                    values,
                })
            }
            SqlStatement::Delete {
                from, selection, ..
            } => {
                let table = if let Some(from_clause) = from.first() {
                    from_clause.relation.to_string()
                } else {
                    return Err(anyhow!("DELETE requires a table name"));
                };

                let where_clause = selection.as_ref().map(|expr| Box::new(expr.clone()));

                Ok(Statement::Delete {
                    table,
                    where_clause,
                })
            }
            SqlStatement::Update {
                table,
                assignments,
                selection,
                ..
            } => {
                let table_name = table.to_string();

                let assignment_pairs = assignments
                    .iter()
                    .map(|assign| {
                        let col_name = assign.id.iter().map(|i| i.value.clone()).collect::<Vec<_>>().join(".");
                        let value = Self::convert_expr(&assign.value)?;
                        Ok((col_name, value))
                    })
                    .collect::<Result<Vec<_>>>()?;

                let where_clause = selection.as_ref().map(|expr| Box::new(expr.clone()));

                Ok(Statement::Update {
                    table: table_name,
                    assignments: assignment_pairs,
                    where_clause,
                })
            }
            SqlStatement::Query(query) => {
                if let ast::SetExpr::Select(select) = &*query.body {
                    let table =
                        if let Some(ast::TableWithJoins { relation, .. }) = select.from.first() {
                            if let ast::TableFactor::Table { name, .. } = relation {
                                name.to_string()
                            } else {
                                return Err(anyhow!("Unsupported table reference"));
                            }
                        } else {
                            return Err(anyhow!("No table specified"));
                        };

                    let columns = select
                        .projection
                        .iter()
                        .map(|proj| match proj {
                            ast::SelectItem::Wildcard(_) => "*".to_string(),
                            ast::SelectItem::UnnamedExpr(expr) => format!("{}", expr),
                            ast::SelectItem::ExprWithAlias { expr: _, alias } => alias.to_string(),
                            _ => "unknown".to_string(),
                        })
                        .collect();

                    let where_clause = select.selection.as_ref().map(|expr| Box::new(expr.clone()));

                    let order_by = if !query.order_by.is_empty() {
                        Some(
                            query
                                .order_by
                                .iter()
                                .map(|order| {
                                    let column = format!("{}", order.expr);
                                    let ascending = order.asc.unwrap_or(true);
                                    crate::sql::types::OrderByClause { column, ascending }
                                })
                                .collect(),
                        )
                    } else {
                        None
                    };

                    let limit = query.limit.as_ref().and_then(|limit_expr| {
                        if let ast::Expr::Value(ast::Value::Number(n, _)) = limit_expr {
                            n.parse().ok()
                        } else {
                            None
                        }
                    });

                    Ok(Statement::Select {
                        table,
                        columns,
                        where_clause,
                        order_by,
                        limit,
                    })
                } else {
                    Err(anyhow!("Unsupported query type"))
                }
            }
            _ => Err(anyhow!("Unsupported statement type")),
        }
    }

    fn convert_column(col: &ColumnDef) -> Result<Column> {
        let name = col.name.to_string();
        let data_type = Self::convert_data_type(&col.data_type)?;
        Ok(Column { name, data_type })
    }

    fn convert_data_type(data_type: &SqlDataType) -> Result<DataType> {
        match data_type {
            SqlDataType::Int(_) | SqlDataType::Integer(_) | SqlDataType::BigInt(_) => {
                Ok(DataType::Integer)
            }
            SqlDataType::Float(_) | SqlDataType::Double | SqlDataType::Real => Ok(DataType::Float),
            SqlDataType::Text
            | SqlDataType::Varchar(_)
            | SqlDataType::Char(_)
            | SqlDataType::String(_) => Ok(DataType::Text),
            SqlDataType::Boolean => Ok(DataType::Boolean),
            _ => Err(anyhow!("Unsupported data type: {:?}", data_type)),
        }
    }

    fn convert_expr(expr: &Expr) -> Result<Value> {
        match expr {
            Expr::Value(ast::Value::Number(n, _)) => {
                if n.contains('.') {
                    Ok(Value::Float(n.parse()?))
                } else {
                    Ok(Value::Integer(n.parse()?))
                }
            }
            Expr::Value(ast::Value::SingleQuotedString(s))
            | Expr::Value(ast::Value::DoubleQuotedString(s)) => Ok(Value::Text(s.clone())),
            Expr::Value(ast::Value::Boolean(b)) => Ok(Value::Boolean(*b)),
            Expr::Value(ast::Value::Null) => Ok(Value::Null),
            _ => Err(anyhow!("Unsupported expression: {:?}", expr)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_create_table() {
        let sql = "CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)";
        let stmt = Parser::parse(sql).unwrap();

        match stmt {
            Statement::CreateTable { name, columns } => {
                assert_eq!(name, "users");
                assert_eq!(columns.len(), 3);
                assert_eq!(columns[0].name, "id");
                assert_eq!(columns[1].name, "name");
            }
            _ => panic!("Expected CreateTable statement"),
        }
    }

    #[test]
    fn test_parse_insert() {
        let sql = "INSERT INTO users (id, name) VALUES (1, 'Alice'), (2, 'Bob')";
        let stmt = Parser::parse(sql).unwrap();

        match stmt {
            Statement::Insert {
                table,
                columns,
                values,
            } => {
                assert_eq!(table, "users");
                assert_eq!(columns.len(), 2);
                assert_eq!(values.len(), 2);
            }
            _ => panic!("Expected Insert statement"),
        }
    }

    #[test]
    fn test_parse_select() {
        let sql = "SELECT id, name FROM users";
        let stmt = Parser::parse(sql).unwrap();

        match stmt {
            Statement::Select { table, columns, .. } => {
                assert_eq!(table, "users");
                assert_eq!(columns.len(), 2);
            }
            _ => panic!("Expected Select statement"),
        }
    }

    #[test]
    fn test_parse_delete_with_where() {
        let sql = "DELETE FROM users WHERE id = 1";
        let stmt = Parser::parse(sql).unwrap();

        match stmt {
            Statement::Delete {
                table,
                where_clause,
            } => {
                assert_eq!(table, "users");
                assert!(where_clause.is_some());
            }
            _ => panic!("Expected Delete statement"),
        }
    }

    #[test]
    fn test_parse_delete_without_where() {
        let sql = "DELETE FROM users";
        let stmt = Parser::parse(sql).unwrap();

        match stmt {
            Statement::Delete {
                table,
                where_clause,
            } => {
                assert_eq!(table, "users");
                assert!(where_clause.is_none());
            }
            _ => panic!("Expected Delete statement"),
        }
    }

    #[test]
    fn test_parse_update_single_column() {
        let sql = "UPDATE users SET name = 'Bob' WHERE id = 1";
        let stmt = Parser::parse(sql).unwrap();

        match stmt {
            Statement::Update {
                table,
                assignments,
                where_clause,
            } => {
                assert_eq!(table, "users");
                assert_eq!(assignments.len(), 1);
                assert_eq!(assignments[0].0, "name");
                assert!(where_clause.is_some());
            }
            _ => panic!("Expected Update statement"),
        }
    }

    #[test]
    fn test_parse_update_multiple_columns() {
        let sql = "UPDATE users SET name = 'Bob', age = 30 WHERE id = 1";
        let stmt = Parser::parse(sql).unwrap();

        match stmt {
            Statement::Update {
                table,
                assignments,
                where_clause,
            } => {
                assert_eq!(table, "users");
                assert_eq!(assignments.len(), 2);
                assert_eq!(assignments[0].0, "name");
                assert_eq!(assignments[1].0, "age");
                assert!(where_clause.is_some());
            }
            _ => panic!("Expected Update statement"),
        }
    }

    #[test]
    fn test_parse_update_without_where() {
        let sql = "UPDATE users SET active = true";
        let stmt = Parser::parse(sql).unwrap();

        match stmt {
            Statement::Update {
                table,
                assignments,
                where_clause,
            } => {
                assert_eq!(table, "users");
                assert_eq!(assignments.len(), 1);
                assert!(where_clause.is_none());
            }
            _ => panic!("Expected Update statement"),
        }
    }
}
