"""Validate canonical/coherent adapter, then rerun requested full groups.
All new outputs contain v2. Original caches are read-only.
"""
import csv
import json
import argparse
import numpy as np
import run_pilot as rp
from config import RESULTS, PHOTON_STATE_COUNTS, LAMBDA_MAGNITUDES, SYSTEMS
from coherent_v2 import displacement


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows)


def save(owner,meta,elapsed):
    rec=rp.cache_calc(owner,meta,elapsed)
    rec['canonical_orbital_coefficients']=np.asarray(owner.C).tolist()
    if getattr(owner,'benchmark_photon_basis','number')=='coherent':
        rec['displacement_alpha']=displacement(owner)
    with (RESULTS/'coherent_v2_cache.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    old=[json.loads(l) for l in (RESULTS/'full_cache.jsonl').read_text().splitlines()]
    def reference(system,geometry,lam):
        return next(r for r in old if r['system']==system and r['geometry']==geometry and r['state']=='photon_convergence' and np.isclose(np.linalg.norm(r['lambda_vector']),lam))
    checks=[]
    mol=rp.geom_variant('H2');basis,ne,no,fc,_=rp.selected_feasibility('H2')
    bare,_=rp.run_calc(mol,basis,'fci',ne,no,0,(0,0,0),1)
    ref=reference('H2','equilibrium',.1)
    norms=np.linalg.norm(np.asarray(bare.electronic_dipole_array)[0],axis=1)
    bright=next(i for i in range(1,len(norms)) if norms[i]>.05)
    outputs={}
    for photon_basis in ('number','coherent'):
        for ns in (6,10):
            owner,t=rp.run_calc(mol,basis,'fci',ne,no,ref['omega'],ref['lambda_vector'],ns,photon_basis=photon_basis)
            lp,up,_=rp.polariton_roots(owner,rp.root_vector(bare,0),rp.root_vector(bare,bright))
            outputs[photon_basis,ns]=(owner,[0,lp,up])
            save(owner,dict(system='H2',geometry='equilibrium',method='fci',basis=basis,active_electrons=ne,active_orbitals=no,state='basis_validation',omega=ref['omega'],lambda_vector=ref['lambda_vector']),t)
            dc=float(np.max(np.abs(owner.C-bare.C)))
            checks.append(dict(check='canonical_orbitals',photon_basis=photon_basis,photon_states=ns,max_C_diff=dc,passed=dc<1e-10))
    for label,k in (('ground',0),('LP',1),('UP',2)):
        for kind,akey,bkey in [('basis_agreement',('number',10),('coherent',10)),('number_truncation',('number',6),('number',10)),('coherent_truncation',('coherent',6),('coherent',10))]:
            a,ar=outputs[akey];b,br=outputs[bkey]
            de=float(abs(a.CIeigs[ar[k]]-b.CIeigs[br[k]]));dr=float(np.linalg.norm(rp.rdm(a,ar[k])-rp.rdm(b,br[k])))
            checks.append(dict(check=kind,state=label,energy_abs_diff=de,rdm_frobenius=dr,passed=de<=1e-6 and dr<=1e-6))
    write_csv(RESULTS/'coherent_v2_validation.csv',checks)
    if not all(c['passed'] for c in checks):raise RuntimeError('H2 validation failed; production blocked')
    print('H2 VALIDATION PASSED',flush=True)
    if args.validate_only:return
    def existing(name):
        path=RESULTS/f'coherent_v2_{name}.csv'
        return list(csv.DictReader(path.open())) if path.exists() else []
    metrics=existing('metrics');convergence=existing('convergence');groups=existing('groups')
    completed={(r['system'],r['geometry'],round(float(r['lambda_value']),4)) for r in groups}
    for geometry,system,scale in rp.full_geometry_cases():
        if system not in {'BeH2','LiH','Li2'}:continue
        if all((system,geometry,l) in completed for l in LAMBDA_MAGNITUDES):continue
        mol=rp.geom_variant(system,scale);basis,ne,no,fc,_=rp.selected_feasibility(system)
        method='casci' if fc else 'fci'
        bare,_=rp.run_calc(mol,basis,method,ne,no,0,(0,0,0),1)
        norms=np.linalg.norm(np.asarray(bare.electronic_dipole_array)[0],axis=1)
        candidates=[i for i in range(1,len(norms)) if norms[i]>.05]
        if not candidates:
            bare,_=rp.run_calc(mol,basis,method,ne,no,0,(0,0,0),1,nroots=24)
            norms=np.linalg.norm(np.asarray(bare.electronic_dipole_array)[0],axis=1)
            candidates=[i for i in range(1,len(norms)) if norms[i]>.05]
        bright=candidates[0]
        gv,bv=rp.root_vector(bare,0),rp.root_vector(bare,bright)
        specs=SYSTEMS[system];ae=specs['valence_electrons']
        spaces=[(ae,o) for o in (specs['valence_orbitals']+x for x in (0,2,4)) if o<no and ae<=ne]
        casbare={}
        for ae,ao in spaces:casbare[ae,ao],_=rp.run_calc(mol,basis,'casci',ae,ao,0,(0,0,0),1)
        for lam in LAMBDA_MAGNITUDES:
            if (system,geometry,lam) in completed:continue
            ref=reference(system,geometry,lam);omega,vec=ref['omega'],ref['lambda_vector']
            previous=None;passed=False;ground_prev=None
            for ns in PHOTON_STATE_COUNTS:
                owner,t=rp.run_calc(mol,basis,method,ne,no,omega,vec,ns,photon_basis='coherent')
                assert np.max(np.abs(owner.C-bare.C))<1e-10,'Electronic orbitals changed'
                lp,up,scores=rp.polariton_roots(owner,gv,bv)
                energies=np.asarray(owner.CIeigs)[[0,lp,up]]
                ambiguous=min(scores[lp],scores[up])<.1
                delta=None if previous is None else float(np.max(np.abs(energies-previous)))
                ci_ok=bool(owner.result.convergence['ci_final'].get('converged'))
                passed=not ambiguous and delta is not None and delta<rp.ENERGY_TOL and ci_ok
                convergence.append(dict(system=system,geometry=geometry,lambda_value=lam,photon_basis='coherent_state',photon_states=ns,ground_energy=energies[0],lp_energy=energies[1],up_energy=energies[2],lp_root=lp,up_root=up,ground_energy_change='' if ground_prev is None else abs(energies[0]-ground_prev),max_target_energy_change='' if delta is None else delta,lp_score=scores[lp],up_score=scores[up],assignment_ambiguous=bool(ambiguous),ci_converged=ci_ok,converged=passed))
                save(owner,dict(system=system,geometry=geometry,method='fci',basis=basis,active_electrons=ne,active_orbitals=no,state='photon_convergence',lambda_vector=vec,omega=omega,lp_root=lp,up_root=up,lp_assignment_score=float(scores[lp]),up_assignment_score=float(scores[up]),assignment_ambiguous=bool(ambiguous)),t)
                print(system,geometry,lam,ns,'delta',delta,'ambiguous',ambiguous,'passed',passed,flush=True)
                previous=None if ambiguous or not ci_ok else energies;ground_prev=energies[0]
                if passed:break
            groups.append(dict(system=system,geometry=geometry,lambda_value=lam,photon_basis='coherent_state',photon_states=ns,converged=passed,reason='converged' if passed else 'ambiguous_polariton_assignment' if ambiguous else 'ci_not_converged' if not ci_ok else 'assignment_gate_no_valid_previous' if delta is None else 'photon_truncation'))
            for ae,ao in spaces:
                qcas,t=rp.run_calc(mol,basis,'casci',ae,ao,omega,vec,ns,photon_basis='coherent')
                assert np.max(np.abs(qcas.C-bare.C))<1e-10
                cas_ok=bool(qcas.result.convergence['ci_final'].get('converged'))
                gc,gf=rp.rdm(qcas,0),rp.rdm(owner,0);qerr,qno=rp.rdm_metrics(gc,gf)
                cerr,cno=rp.rdm_metrics(rp.rdm(casbare[ae,ao],0),rp.rdm(bare,0))
                metrics.append(dict(system=system,geometry=geometry,basis=basis,active_electrons=ae,active_orbitals=ao,active_space=f'CAS({ae},{ao})',state='ground',condition='resonant',lambda_value=lam,photon_basis='coherent_state',electronic_orbitals='canonical_cavity_free_rhf',frozen_core_orbitals=int(qcas.n_in_a),reference_frozen_core_orbitals=fc,omega=omega,cavity_free_frobenius=cerr,qed_frobenius=qerr,ratio=qerr/cerr,cavity_free_natural_occupation_error=cno,qed_natural_occupation_error=qno,natural_occupation_ratio=qno/cno,energy=float(qcas.CIeigs[0]),photon_number='',trace=float(np.trace(gc)),photon_states=ns,timing_s=t,assignment_ambiguous=False,status='converged' if passed and cas_ok else 'not_converged_at_max'))
                save(qcas,dict(system=system,geometry=geometry,method='casci',basis=basis,active_electrons=ae,active_orbitals=ao,state='ground',lambda_vector=vec,omega=omega),t)
            write_csv(RESULTS/'coherent_v2_metrics.csv',metrics)
            write_csv(RESULTS/'coherent_v2_convergence.csv',convergence)
            write_csv(RESULTS/'coherent_v2_groups.csv',groups)
    print('PRODUCTION COMPLETE',flush=True)

if __name__=='__main__':main()
