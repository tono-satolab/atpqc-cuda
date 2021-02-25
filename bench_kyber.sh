#! /bin/bash -e

##
## bench_kyber.sh
## Executes benchmark and outputs results.
##
## Copyright (c) 2021 Tatsuki Ono
##
## This software is released under the MIT License.
## https://opensource.org/licenses/mit-license.php
##

bld_dir=./target
out_dir=${bld_dir}/bench

if [ ! -d ${out_dir} ]
then
	mkdir -p ${out_dir}
fi

make clean
make bench

for variant in 512 768 1024
do
	fname="bench_kyber${variant}"
	rm -fv ${out_dir}/${fname}_result.txt && touch ${out_dir}/${fname}_result.txt

	for ((ninput=1 ; ninput<32769 ; ninput=ninput*2))
	do
		for ((genmat_nwarps=1 ; genmat_nwarps<32 ; genmat_nwarps=genmat_nwarps*2))
		do
			for ((genvec_nwarps=1 ; genvec_nwarps<32 ; genvec_nwarps=genvec_nwarps*2))
			do
				for ((genpoly_nwarps=1 ; genpoly_nwarps<32 ; genpoly_nwarps=genpoly_nwarps*2))
				do
					for ((keccak_nwarps=1 ; keccak_nwarps<32 ; keccak_nwarps=keccak_nwarps*2))
					do
						echo "${ninput} ${genmat_nwarps} ${genvec_nwarps} ${genpoly_nwarps} ${keccak_nwarps}"
						echo -n "${ninput},${genmat_nwarps},${genvec_nwarps},${genpoly_nwarps},${keccak_nwarps}," >> ${out_dir}/${fname}_result.txt
						echo "${ninput} ${genmat_nwarps} ${genvec_nwarps} ${genpoly_nwarps} ${keccak_nwarps}" | ${bld_dir}/${fname}.out >> ${out_dir}/${fname}_result.txt
					done
				done
			done
		done
	done
done
