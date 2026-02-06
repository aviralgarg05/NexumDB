# NexumDB Roadmap

This document outlines the planned features and improvements for NexumDB.

## Legend
- **Planned** - On the roadmap
- **In Progress** - Currently being worked on
- **Completed** - Shipped in a release
- **Considering** - Under evaluation

---

## v0.5.0 (Next Release)
**In Progress: Open Source Readiness & Community Building**

### Infrastructure (Completed)
- [x] Complete OSS setup (LICENSE, CONTRIBUTING, CODE_OF_CONDUCT)
- [x] Comprehensive CI/CD pipelines
- [x] Security scanning (CodeQL, cargo audit)
- [x] Automated releases (Release Please)
- [x] Coverage tracking (Codecov)
- [x] Auto-labeling for PRs
- [x] Welcome workflow for contributors
- [x] Semantic PR title checking
- [x] PR size labeling
- [x] Dependabot auto-merge

---

## v0.6.0 - Enhanced SQL Features
**Planned Q1 2026**

### SQL Improvements
- [ ] JOIN operations (INNER, LEFT, RIGHT, FULL)
- [ ] Subqueries support
- [ ] Aggregate functions (SUM, AVG, COUNT, MIN, MAX)
- [ ] GROUP BY and HAVING clauses
- [ ] DISTINCT keyword
- [ ] UNION, INTERSECT, EXCEPT operations

### Performance
- [ ] Query optimization hints
- [ ] Index support (B-Tree, Hash)
- [ ] Query plan visualization
- [ ] Parallel query execution

---

## v0.7.0 - Advanced AI Features
**Planned Q2 2026**

### AI/ML Enhancements
- [ ] Multi-model support (different LLMs)
- [ ] Fine-tuning on user queries
- [ ] Anomaly detection in data
- [ ] Predictive analytics integration
- [ ] Auto-suggest query completions
- [ ] Query pattern learning

### Semantic Features
- [ ] Vector similarity search
- [ ] Semantic data clustering
- [ ] Intelligent data deduplication
- [ ] Context-aware caching improvements

---

## v0.8.0 - Enterprise Features
**Planned Q3 2026**

### Security & Compliance
- [ ] Row-level security
- [ ] Column-level encryption
- [ ] Audit logging
- [ ] RBAC (Role-Based Access Control)
- [ ] LDAP/OAuth integration
- [ ] Compliance reports (SOC2, GDPR)

### Scalability
- [ ] Distributed storage backend
- [ ] Replication support
- [ ] Sharding capabilities
- [ ] Load balancing
- [ ] Cluster management

---

## v0.9.0 - Developer Experience
**Planned Q4 2026**

### Tooling
- [ ] VS Code extension
- [ ] Language Server Protocol (LSP)
- [ ] GUI admin interface
- [ ] Migration tool from other databases
- [ ] Schema designer
- [ ] Query builder UI

### SDKs
- [ ] Python SDK
- [ ] JavaScript/TypeScript SDK
- [ ] Go SDK
- [ ] Java SDK
- [ ] REST API
- [ ] GraphQL API

---

## v1.0.0 - Production Ready
**Target: Mid 2027**

### Stability
- [ ] Comprehensive test coverage (>90%)
- [ ] Performance benchmarks vs. competitors
- [ ] Production deployment guides
- [ ] SLA guarantees
- [ ] Professional support options

### Documentation
- [ ] Complete API documentation
- [ ] Video tutorials
- [ ] Migration guides
- [ ] Best practices guide
- [ ] Architecture deep-dives

---

## Future Considerations 💭

### Integration Ideas
- Cloud provider integrations (AWS, Azure, GCP)
- Data lake connectors
- Stream processing (Kafka, Kinesis)
- Time-series optimization
- Graph database capabilities
- Blockchain/ledger features

### Community Features
- Plugin system
- Custom function registration
- Extension marketplace
- Community voting on features
- Bounty program

---

## How to Contribute

We welcome contributions to any of these features! Here's how:

1. **Check existing issues**: See if someone is already working on it
2. **Create a discussion**: Propose your approach
3. **Submit a PR**: Follow our contributing guidelines
4. **Get reviewed**: Collaborate with maintainers

Priority areas for contributions:
- 🟢 **Good First Issue**: Great for new contributors
- 🟡 **Help Wanted**: We'd love community help!
- 🔴 **Core Team Focus**: We're working on this

---

## Version History

### v0.4.0 (Current)
- Project-correct SELECT with column/alias projection
- Schema-safe writes (INSERT/UPDATE validation)
- Table management (SHOW TABLES, DESCRIBE, DROP TABLE)
- Cache safety improvements

### v0.3.0
- Advanced SQL operators (LIKE, IN, BETWEEN)
- Query modifiers (ORDER BY, LIMIT)
- Persistent RL agent
- Model management

### v0.2.0
- WHERE clause filtering
- Natural language queries (ASK)
- Reinforcement learning
- Expression evaluator

### v0.1.0
- Initial SQL support
- Semantic caching
- Self-optimizing execution
- Local-only operation

---

*This roadmap is subject to change based on community feedback and priorities. Last updated: February 6, 2026*
