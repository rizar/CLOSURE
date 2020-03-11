function gen {
    T=$1
    shift
    python ../../scripts/generate_templates.py ${T}_meta.json ${T}.json $@
    echo $T `jq length ${T}.json`
} 

gen chain_sim_loc
gen chain_loc_sim

gen compare_sim --query-material=0
gen compare_sim_loc --query-material=0

gen or_sim
gen or_sim_loc

gen and_sim_loc
