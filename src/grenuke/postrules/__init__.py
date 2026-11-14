"""Fraud-block post rules on decoded October labels (label-free; class sets fitted on train labels).

`rules.py` holds the rules "order" (class-order feasibility) and "mulebot" (mule|bot edge), and
`rules_forced.py` the rule "order-forced" (order feasibility only where exactly one class is
feasible). `tag_class_counts.json` holds the per-shape class counts of fraud-labelled train rows the
class sets are fitted on. `rules34.py` holds the generator-spec rules 4a (short chain) and 4b
(merchant QR), with the merchant-QR state's train class counts in `state_class_counts.json`; its
rule 3 "recut" is not used by any config. Used by `grenuke.post.block_rule` and
`grenuke.post.spec_rule`."""
