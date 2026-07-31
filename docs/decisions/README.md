# Architecture Decision Records

ADR 记录会长期影响复现性、协议身份、模块边界或结果解释的设计取舍。普通实现细节、一次性修复和
运行状态不写 ADR。

## 索引

| ID | 状态 | 决策 |
|---|---|---|
| [0001](0001-separate-inference-and-scorer-protocols.md) | Accepted | 分离 inference protocol 与 scorer/cache protocol |

## 新 ADR 触发条件

出现以下任一情况时，在实现完成前新增或 supersede ADR：

- 改变 benchmark/model/shell 的职责边界；
- 改变协议或 cache identity 的划分方法；
- 选择与官方实现不同且会影响结果解释的长期方案；
- 引入新的正式结果目录、状态机或发布门禁；
- 推翻现有 ADR。

同一决策只维护一个当前 ADR。被替代的 ADR 保留历史，但标为 `Superseded by ADR-XXXX`。

## 模板

```markdown
# ADR-XXXX: 标题

- Status: Proposed | Accepted | Superseded
- Date: YYYY-MM-DD
- Supersedes: none | ADR-XXXX

## Context

需要解决的问题、协议约束和备选方案。

## Decision

最终选择及其明确边界。

## Consequences

正面影响、代价、迁移要求和必须新增的测试/文档。

## References

相关代码、协议、测试和 commit。
```
