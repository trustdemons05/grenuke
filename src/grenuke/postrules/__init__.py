"""Fraud-block post rules on decoded October labels (label-free; class sets fitted on train labels).

`rules.py` holds the rules "order" (class-order feasibility) and "mulebot" (mule|bot edge), and
`rules_forced.py` the rule "order-forced" (order feasibility only where exactly one class is
feasible). `tag_class_counts.json` holds the per-shape class counts of fraud-labelled train rows the
class sets are fitted on. `rules34.py` holds the generator-spec rules 4a (short chain) and 4b
(merchant QR), with the merchant-QR state's train class counts in `state_class_counts.json`; its
rule 3 "recut" is not used by any config. `edges.py` holds the rule "edge": a chain model of the
block's class-zone borders, with rates fitted on the train block, that re-types fraud calls on the
unlikely side of the borders a config lists. `bursts.py` holds the rule "bot burst": mule bursts
with no block phone, above a row count and below a sender SIM age that a config sets, re-typed to
velocity_bot. Used by `grenuke.post.block_rule`, `grenuke.post.edge_rule`,
`grenuke.post.spec_rule` and `grenuke.post.burst_rule`."""
