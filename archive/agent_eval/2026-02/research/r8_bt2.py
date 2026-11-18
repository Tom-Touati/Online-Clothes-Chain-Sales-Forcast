import sys; sys.path.insert(0, 'research'); sys.path.insert(0, '.')
from r4_bt import run
import copy
run('v13 (tags+June, Sweatpants)')
run('no type trend', {'TYPE_TREND': {}})
run('Sweatpants 1.04', {'TYPE_TREND': {'Sweatpants': 1.04}})
run('W14 0.5', {'W14': 0.5})
def oldtags(r):
    r.OCCASION["4th of July"][6] = 1.3; r.OCCASION["Graduation"][6] = 1.0
run('4th Jun 1.3 / Grad Jun 1.0', mod=oldtags)
