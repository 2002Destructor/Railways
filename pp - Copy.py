import json, os
from dataclasses import dataclass
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pandas as pd
import numpy as np

# ============================================================
# PARAMETERS (MATCH 2milp)
# ============================================================

class Param:
    BIG_M = 2000
    MIN_HEADWAY = 3.0

    TURN_LB = 5.0
    TURN_UB = 10.0

    RAKE_WEIGHT = 10000.0
    LINK_REWARD = 2000.0
    DELTA_WEIGHT = 1.0

    TIME_LIMIT = 600
    OUTPUT_DIR = "outputs_json"

    EVENT_COLS = [
        'CCGa','CCGd','MCTa','MCTd','DDRa','DDRd','BAa','BAd','ADHa','ADHd',
        'GMNa','GMNd','BVIa','BVId','BYRa','BYRd','BSRa','BSRd','VRa','VRd',
        'DRDa','DRDd'
    ]


# ============================================================
# SERVICE
# ============================================================

@dataclass
class Service:
    start: float
    typ: str
    direction: str
    frm: str
    to: str
    segments: dict
    first_dep: int
    last_arr: int


# ============================================================
# JSON LOADER
# ============================================================

class JsonLoader:

    def __init__(self,fname):
        with open(fname) as f:
            self.J=json.load(f)

    def services(self):
        out=[]
        for v in self.J["services"].values():
            out.append(Service(
                start=v["start_time"],
                typ=v["Type"],
                direction=v["Dir"],
                frm=v["From"],
                to=v["To"],
                segments=v["segments"],
                first_dep=int(v["first_dep"]),
                last_arr=int(v["last_arr"])
            ))
        return out

    def events(self):
        ev=set()
        for s in self.J["services"].values():
            ev.add(int(s["first_dep"]))
            ev.add(int(s["last_arr"]))
            for k in s["segments"]:
                a,b=k.split("_")
                ev.add(int(a)); ev.add(int(b))
        return sorted(ev)

    # turnaround = (arrival , departure)
    def turnaround_pairs(self):
        P=[]
        T=self.J["turnaround"]

        for st in T:
            # DOWN arr -> UP dep
            for a in T[st]["DOWN"]["arr"]:
                for d in T[st]["UP"]["dep"]:
                    P.append((int(a),int(d)))

            # UP arr -> DOWN dep
            for a in T[st]["UP"]["arr"]:
                for d in T[st]["DOWN"]["dep"]:
                    P.append((int(a),int(d)))
        return P

    def headway_pairs(self):
        P=[]
        for d in self.J["headway"]:
            for typ in self.J["headway"][d]:
                for st,ids in self.J["headway"][d][typ].items():
                    for i in range(len(ids)):
                        for j in range(i+1,len(ids)):
                            P.append((int(ids[i]),int(ids[j])))
        return P

    def distribution(self):
        return self.J["distribution"]


# ============================================================
# SCHEDULER
# ============================================================

class Scheduler:

    def __init__(self,loader):
        self.services=loader.services()
        self.events=loader.events()
        self.turn=loader.turnaround_pairs()
        self.head=loader.headway_pairs()
        self.dist=loader.distribution()
        self.m=pyo.ConcreteModel()

    def build(self):
        m=self.m

        m.EVENTS=pyo.Set(initialize=self.events)
        m.LINKS=pyo.Set(initialize=self.turn,dimen=2)
        m.HW=pyo.Set(initialize=self.head,dimen=2)

        m.t=pyo.Var(m.EVENTS,within=pyo.NonNegativeReals)
        m.delta=pyo.Var(within=pyo.NonNegativeReals)

        m.X=pyo.Var(m.LINKS,within=pyo.Binary)
        m.p=pyo.Var(m.HW,within=pyo.Binary)

        ARR={a for a,_ in self.turn}
        DEP={d for _,d in self.turn}

        m.source=pyo.Var(DEP,within=pyo.Binary)
        m.sink=pyo.Var(ARR,within=pyo.Binary)

        m.trav=pyo.ConstraintList()
        m.headway=pyo.ConstraintList()
        m.rake=pyo.ConstraintList()
        m.distc=pyo.ConstraintList()
        m.start=pyo.ConstraintList()

        # ---- traversal (DIRECT, no chain rebuild)
        for s in self.services:
            for k,tt in s.segments.items():
                a,b=k.split("_")
                a=int(a); b=int(b)
                m.trav.add(m.t[b]-m.t[a]==tt)
                m.trav.add(m.t[b]>=m.t[a])

        # ---- headway
        for i,j in m.HW:
            m.headway.add(m.t[j]-m.t[i]+(Param.BIG_M+Param.MIN_HEADWAY)*m.p[i,j]>=Param.MIN_HEADWAY)
            m.headway.add(m.t[j]-m.t[i]+(Param.BIG_M+Param.MIN_HEADWAY)*m.p[i,j]<=Param.BIG_M)

        # ---- rake conservation
        for j in ARR:
            m.rake.add(sum(m.X[a,j] for a,_ in self.turn if _==j)+m.sink[j]==1)
        for i in DEP:
            m.rake.add(sum(m.X[a,i] for a,_ in self.turn if a==i)+m.source[i]==1)

        m.rake.add(sum(m.source[i] for i in DEP)==sum(m.sink[j] for j in ARR))

        # ---- turnaround bounds
        for a,d in self.turn:
            m.rake.add(m.t[d]-m.t[a]>=Param.TURN_LB-Param.BIG_M*(1-m.X[a,d]))
            m.rake.add(m.t[d]-m.t[a]<=Param.TURN_UB+Param.BIG_M*(1-m.X[a,d]))

        # ---- distribution (same math as 2milp)
        for ids in self.dist.values():
            if len(ids)>1:
                ideal=60.0/(len(ids)+1)
                for u in range(len(ids)-1):
                    a=int(ids[u]); b=int(ids[u+1])
                    m.distc.add(m.t[b]-m.t[a]>=ideal-m.delta)
                    m.distc.add(m.t[b]-m.t[a]<=ideal+m.delta)

        # ---- start windows
        first=True
        for s in sorted(self.services,key=lambda x:x.start):
            if first:
                m.start.add(m.t[s.first_dep]==480)
                first=False
            else:
                m.start.add(m.t[s.first_dep]>=s.start)
                m.start.add(m.t[s.first_dep]<=s.start+19)

        # ---- objective (IDENTICAL)
        m.obj=pyo.Objective(
            expr=
              Param.RAKE_WEIGHT*(sum(m.source[i] for i in DEP)+sum(m.sink[j] for j in ARR))
            - Param.LINK_REWARD*sum(m.X[a,d] for a,d in m.LINKS)
            + Param.DELTA_WEIGHT*m.delta,
            sense=pyo.minimize)

        return m


# ============================================================
# SOLVER + CSV
# ============================================================

class Optimizer:
    def __init__(self,m): self.m=m

    def solve(self):
        s=SolverFactory("cbc")
        s.options["seconds"]=Param.TIME_LIMIT
        return s.solve(self.m,tee=True)

    def times(self):
        return {i:float(self.m.t[i].value) for i in self.m.EVENTS if self.m.t[i].value is not None}


def replace_csv(times,up="1-o-event-ids_UP.csv",down="1-o-event-ids_DOWN.csv"):
    os.makedirs(Param.OUTPUT_DIR,exist_ok=True)

    def run(f,out):
        df=pd.read_csv(f)
        for c in Param.EVENT_COLS:
            if c in df.columns:
                df[c]=df[c].apply(lambda v: round(times.get(int(v),np.nan),2) if pd.notna(v) else np.nan)
        df["Optimized_Start_Time"]=df[Param.EVENT_COLS].min(axis=1)
        df.sort_values("Optimized_Start_Time").to_csv(out,index=False)
        print("Saved",out)

    run(up,f"{Param.OUTPUT_DIR}/optimized_UP.csv")
    run(down,f"{Param.OUTPUT_DIR}/optimized_DOWN.csv")


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(Param.OUTPUT_DIR,exist_ok=True)

    loader=JsonLoader("milp_preprocessed.json")
    model=Scheduler(loader).build()

    res=Optimizer(model).solve()

    if res.solver.termination_condition!=pyo.TerminationCondition.optimal:
        print("Solver:",res.solver.termination_condition)
        return

    times=Optimizer(model).times()
    replace_csv(times)

    print("Solved. Events:",len(times))

if __name__=="__main__":
    main()
