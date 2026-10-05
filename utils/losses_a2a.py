# --*--coding   : utf-8  --*--
# @Time         : 2026/5/6 10:00
# @Author       : RisingInsight
# @File         : losses_a2a.py
# @Software     : PyCharm
# @Project      : OIGA2A-new
# @Function     : Loss factory for A2A (Any-to-Any) pipeline
#
# NOTE: The loss implementations (including the Dynamic Loss Mask Router)
# are withheld from this partial release. The full implementation will be
# released after the paper review is concluded.


def get_loss(loss_name, loss_config):
    """Loss factory for the A2A (Any-to-Any) pipeline.

    Args:
        loss_name: name of the loss family (e.g. "sit").
        loss_config: loss-specific configuration dict.
    """
    raise NotImplementedError("OIGA2A: full implementation will be released upon paper acceptance.")
