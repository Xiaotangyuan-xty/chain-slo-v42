"""Two explicit causal bridges; original C and default h4 stay frozen."""
import collections
from types import SimpleNamespace
from frozen_membership import ARMS as PREVIOUS_ARMS,wrapper as previous_wrapper,default_h4

ARMS=('C_frame_anchor_fair_refill','D_default_h4',
      'E_C_with_h4_expiry','F_D_with_GMAX_shadow')

def make_policy(original,arm,decode_seconds):
    assert arm in ARMS and arm!=ARMS[1]
    h=4*decode_seconds
    original_C=previous_wrapper(original,PREVIOUS_ARMS[2])
    if arm==ARMS[0]:return original_C
    if arm==ARMS[2]:
        class ExpiringFrameAnchor(original_C):
            def __init__(self,*args,**kw):
                super().__init__(*args,**kw)
                self.lease_remaining=[0.,0.]
                self.bridge_counts=collections.Counter()

            def choose(self,view,instance,legal,previous):
                old_anchor=self.frame_anchors[instance]
                old_remaining=self.lease_remaining[instance]
                expired=old_remaining<=0
                if expired:self.frame_anchors[instance]=None
                selected,audit=super().choose(view,instance,legal,previous)
                reason=None
                if audit['official_frame']:reason='official_frame_reset'
                elif old_anchor not in legal:reason='anchor_invalid'
                elif expired:reason='h4_expired_default_reselect'
                if reason:
                    self.lease_remaining[instance]=h
                    self.bridge_counts[reason]+=1
                audit.update(bridge_arm=arm,lease_remaining_before=old_remaining,
                             lease_remaining_at_dispatch=self.lease_remaining[instance],
                             lease_renew_reason=reason,selected_h=h,
                             previous_frame_anchor=old_anchor)
                return selected,audit

            def after_step(self,instance,actually_selected):
                if self.frame_anchors[instance] in actually_selected:
                    self.lease_remaining[instance]-=decode_seconds

            def snapshot(self):
                out=super().snapshot()
                out.update(bridge_arm=arm,bridge_counts=dict(self.bridge_counts))
                return out
        return ExpiringFrameAnchor

    class DefaultWithGMAXShadow(original):
        def __init__(self,*args,**kw):
            super().__init__(*args,**kw)
            self.default_anchors=[None,None]
            self.lease_remaining=[0.,0.]
            self.bridge_counts=collections.Counter()

        def choose(self,view,instance,legal,previous):
            raw,audit=super().choose(view,instance,legal,previous)
            old=self.default_anchors[instance];remaining=self.lease_remaining[instance]
            action_audit={}
            if old not in legal or remaining<=0:
                (key,lease),action_audit=default_h4(view,instance,legal,None,SimpleNamespace(decode_seconds=decode_seconds),None)
                self.default_anchors[instance]=key;self.lease_remaining[instance]=lease
                self.bridge_counts['default_h4_decisions']+=1
            anchor=self.default_anchors[instance]
            fair=sorted((c for c in view.calls if c.key in legal),key=lambda c:(c.level,-c.ready_wait,c.key))
            selected=([anchor]+[c.key for c in fair if c.key!=anchor])[:4]
            self.bridge_counts['shadow_tick']+=1
            self.bridge_counts['shadow_frame']+=bool(audit['official_frame'])
            self.bridge_counts['shadow_would_change_members']+=set(raw)!=set(selected)
            audit.update(bridge_arm=arm,shadow_only=True,raw_official_selection=list(raw),
                         actual_default_selection=list(selected),default_anchor=anchor,
                         lease_remaining_before=remaining,
                         lease_remaining_at_dispatch=self.lease_remaining[instance],
                         default_action_audit=action_audit,selected_h=h)
            return selected,audit

        def after_step(self,instance,actually_selected):
            if self.default_anchors[instance] in actually_selected:
                self.lease_remaining[instance]-=decode_seconds

        def snapshot(self):
            out=super().snapshot()
            out.update(bridge_arm=arm,shadow_only=True,bridge_counts=dict(self.bridge_counts),
                       scope='official prepare/assign/frame/finish execute on actual D state; choices ignored')
            return out
    return DefaultWithGMAXShadow
