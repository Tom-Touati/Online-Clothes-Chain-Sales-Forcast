import sys; sys.path.insert(0,"research"); sys.path.insert(0,".")
import bt
P3=dict(OTHER_MONTHLY=0.9,RET_POW=0.7)
bt.run("v3 (P3, v3 occ)",P3,occ={"4th of July":{6:1.25,7:1.25}},site={})
bt.run("v4 final")
bt.run("v4 + other 0.9",{"OTHER_MONTHLY":0.9})
