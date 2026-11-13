"""Regression tests for lambda selection and panel/statistic membership."""
import numpy as np
from plot_combined import selected,stats
from config import lambda_key


def test_float_roundoff_and_state_selection():
    rows=[dict(status='converged',state=state,lambda_value=str(lam),
        cavity_free_frobenius='0.1',ratio='1.1',bright_dominated=True)
        for state in ('ground','ground_offres','bright_dressed','LP','UP')
        for lam in (.01,np.nextafter(.05,0),np.nextafter(.1,0))]
    a=selected(rows,'cavity_free_frobenius','ratio')
    b=selected(rows,'cavity_free_frobenius','ratio',polaritons=True)
    ratios=selected(rows,'cavity_free_frobenius','ratio',ratio=True,polaritons=True)
    assert len(a)==len(ratios)==6 and len(b)==12
    assert {r['state'] for r in a}=={'ground','bright_dressed'}
    assert [r['n'] for r in stats(ratios,'test')]==[2,2,2]
    assert stats(b,'test')==stats(ratios,'test')
    assert lambda_key(np.nextafter(.1,0))==lambda_key(.1)


def test_ineligible_rows_are_excluded():
    base=dict(status='converged',state='ground',lambda_value=.05,x=.1,y=1.)
    rows=[dict(base,status='not_converged_at_max'),dict(base,assignment_ambiguous=True),
          dict(base,lambda_value=.051),dict(base,state='bright_dressed',bright_dominated=False),
          dict(base,x=0),dict(base,y=float('nan'))]
    assert not selected(rows,'x','y')
