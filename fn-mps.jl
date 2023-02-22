
using ITensors
using Dates

function nor_rdm(σσ::ITensor)

    id_t = inds(σσ)
    id_len = length(id_t)

    id_no_p = []
    i_count = 1
    for id in id_t
        if plev(id) == 0
            append!(id_no_p, i_count)
        end
        i_count += 1
    end

    cont_σ = σσ

    for id in id_no_p
        cont_σ = cont_σ * delta(id_t[id], id_t[id]')
    end

    return σσ / cont_σ

end


function aLRsvd(aC, C, id_l, id_r, id_d)
    # input parameters
    # aC, C : input tensors
    # id_l, id_r, id_d : indicies of input tensors
    # id_lr : temperal index, can be defined newly within the function

    #    id_l - aC - id_r      id_l - C - id_r
    #           |
    #          id_d
    
    Cdag = replaceinds(conj(C), [id_l,id_r], [id_r, id_l]);
    linds = (id_l, id_d);
    uL,~,vL = svd(replaceind(aC, id_r=>id_lr) *replaceind(Cdag, id_l=>id_lr),linds...,);

    idLu = uniqueind(uL,aC);
    idLv = uniqueind(vL,aC);
    aL = uL*replaceind(vL, idLv=>idLu);

    epsiL = norm(aC - replaceind(aL, id_r=>id_lr) * replaceind(C, id_l=>id_lr));

    uR,~,vR = svd(replaceind(Cdag, id_r=>id_lr) *replaceind(aC, id_l=>id_lr) ,id_l);

    idLu = uniqueind(uR,aC);
    idLv = uniqueind(vR,aC);
    aR=uR*replaceind(vR,idLv=>idLu);

    epsiR = norm(aC -   replaceind(C, id_r=>id_lr) * replaceind(aR , id_l=>id_lr));
    epsiPrec = max(epsiL, epsiR);
    return aL, aR, epsiPrec, epsiL, epsiR
end
# array(nor_rdm(σσ))

function randaCC(id_d, id_l, id_r,  db)
    # inputs
    # id_d, id_l, id_r : indicies
    # db = dimensions of bond indices
    rndA = randomITensor(ComplexF64, id_d, id_l, id_r);
    rndA = rndA + replaceinds(rndA,[id_l, id_r], [id_r, id_l]);
    rndA = rndA / norm(rndA);

    rndC = diagITensor(sort(rand(db), rev=true), id_l, id_r);
    rndC = rndC/ norm(rndC);

    # outputs
    #    id_l - aC - id_r      id_l - C - id_r
    #           |
    #          id_d

    return rndA, rndC

end

function randaCC_2(id_d, id_l, id_r,  db)
    # inputs
    # id_d, id_l, id_r : indicies
    # db = dimensions of bond indices
    rndA = randomITensor(ComplexF64, id_d, id_l, id_r);
    rndA = rndA / norm(rndA);

    # rndC = diagITensor(sort(rand(db), rev=true), id_l, id_r);
    rndC = randomITensor(ComplexF64, id_l, id_r);
    rndC = rndC/ norm(rndC);

    # outputs
    #    id_l - aC - id_r      id_l - C - id_r
    #           |
    #          id_d

    return rndA, rndC

end

function calLR(id_l, id_r, tL, tR, epsiPrec, shft = -1)
    ## Calculate R : itR
    ## transfer tL into rank-2 tensor
    # generate combiners and combined indicies
    cbL = combiner(id_l, id_l', tags="cbL");
    idcbL = uniqueind(cbL, tL);
    cbR = combiner(id_r, id_r', tags="cbR");
    idcbR = uniqueind(cbR, tL);
    # tolerance of power method
    ptol = min(1e-3, epsiPrec)/100;

    # calculate combined ITensor
    cbtL = cbL * tL * cbR;

    ## Calculate right eigenstate of tL with iteratitive power method
    A = matrix(cbtL) - shft*I;
    ~, psR = powm(A,     inverse = false, shift = shft, tol = ptol, maxiter = 10000 );
    #        powm(A-σ*I, inverse = false, shift = σ,    tol = 1e-4, maxiter = 200);
    # psR = psR * norm(psR[1])/psR[1];  # normalize phase

    ## transfer the form of the right eigenstate into ITensor 
    itR = ITensor(psR, idcbR);
    itR = itR * dag(cbR); itR = itR / (array(itR * delta(id_r, id_r'))[1]);
    itIL = diagITensor(ComplexF64,1, id_l, id_l');

    # eigenvector error check 
    ec_diag_tL_L = norm(array(replaceinds(tL*itIL, [id_r, id_r'],[id_l, id_l']) - itIL));
    ec_diag_tL_R = norm(array(replaceinds(tL * itR, [id_l, id_l'],[id_r, id_r']) -  itR));


    ## Calculate L : itL
    cbtR = cbR * tR * cbL;
    ## Calculate right eigenstate of tR with iteratitive power method
    shft = -1.0;
    A = matrix(cbtR) - shft*I;
    ~, psL = powm(A,     inverse = false, shift = shft, tol = ptol, maxiter = 10000 );
    #        powm(A-σ*I, inverse = false, shift = σ,    tol = 1e-4, maxiter = 200);

    # psL = psL * norm(psL[1])/psL[1];  # normalize phase

    ## transfer the form of the left eigenstate into ITensor 
    itL = ITensor(psL, idcbL);
    itL = itL * dag(cbL); itL = itL / (array(itL* delta(id_l, id_l'))[1] );
    itIR = diagITensor(ComplexF64,1, id_r, id_r');

    # eigenvector error check 
    ec_diag_tR_R = norm(array(replaceinds(tR * itIR,[id_l, id_l'],[id_r, id_r']) - itIR));
    ec_diag_tR_L = norm(array(replaceinds(tR * itL, [id_r, id_r'],[id_l, id_l']) -  itL));

    if (ec_diag_tL_L <10* ptol) &   (ec_diag_tL_R <10* ptol) & (ec_diag_tR_R < 10* ptol) & (ec_diag_tR_L < 10*ptol)
        1+1
    else
        println("calLR: error check: ptol = ", ptol, "\n")
        println("ec_diag_tL_L = ",ec_diag_tL_L," ec_diag_tL_R = ", ec_diag_tL_R,
        " ec_diag_tR_R = ",ec_diag_tR_R, " ec_diag_tR_L = ", ec_diag_tR_L)
    end
    return itR, itL, itIL, itIR
    # , ec_diag_tL_L, ec_diag_tL_R, ec_diag_tR_R, ec_diag_tR_L
end


# function calLR(id_l, id_r, tL, tR, epsiPrec, shft = -1)
#     ## Calculate R : itR
#     ## transfer tL into rank-2 tensor
#     # generate combiners and combined indicies
#     cbL = combiner(id_l, id_l', tags="cbL");
#     idcbL = uniqueind(cbL, tL);
#     cbR = combiner(id_r, id_r', tags="cbR");
#     idcbR = uniqueind(cbR, tL);
#     # tolerance of power method
#     ptol = min(1e-3, epsiPrec)/100;

#     # calculate combined ITensor
#     cbtL = cbL * tL * cbR;

#     ## Calculate right eigenstate of tL with iteratitive power method
#     A = matrix(cbtL) - shft*I;
#     ~, psR = powm(A,     inverse = false, shift = shft, tol = ptol, maxiter = 10000 );
#     #        powm(A-σ*I, inverse = false, shift = σ,    tol = 1e-4, maxiter = 200);
#     psR = psR * norm(psR[1])/psR[1]; psR = psR/norm(psR); # normalize size and phase

#     ## transfer the form of the right eigenstate into ITensor 
#     itR = ITensor(psR, idcbR);
#     itR = itR * dag(cbR); 
#     itIL = diagITensor(ComplexF64,1, id_l, id_l');

#     # eigenvector error check 
#     ec_diag_tL_L = norm(array(replaceinds(tL*itIL, [id_r, id_r'],[id_l, id_l']) - itIL));
#     ec_diag_tL_R = norm(array(replaceinds(tL * itR, [id_l, id_l'],[id_r, id_r']) -  itR));


#     ## Calculate L : itL
#     cbtR = cbR * tR * cbL;
#     ## Calculate right eigenstate of tR with iteratitive power method
#     shft = -1.0;
#     A = matrix(cbtR) - shft*I;
#     ~, psL = powm(A,     inverse = false, shift = shft, tol = ptol, maxiter = 10000 );
#     #        powm(A-σ*I, inverse = false, shift = σ,    tol = 1e-4, maxiter = 200);

#     psL = psL * norm(psL[1])/psL[1]; psL = psL/norm(psL); # normalize size and phase

#     ## transfer the form of the left eigenstate into ITensor 
#     itL = ITensor(psL, idcbL);
#     itL = itL * dag(cbL); 
#     itIR = diagITensor(ComplexF64,1, id_r, id_r');

#     # eigenvector error check 
#     ec_diag_tR_R = norm(array(replaceinds(tR * itIR,[id_l, id_l'],[id_r, id_r']) - itIR));
#     ec_diag_tR_L = norm(array(replaceinds(tR * itL, [id_r, id_r'],[id_l, id_l']) -  itL));

#     if (ec_diag_tL_L <10* ptol) &   (ec_diag_tL_R <10* ptol) & (ec_diag_tR_R < 10* ptol) & (ec_diag_tR_L < 10*ptol)
#         1+1
#     else
#         println("calLR: error check: ptol = ", ptol, "\n")
#         println("ec_diag_tL_L = ",ec_diag_tL_L," ec_diag_tL_R = ", ec_diag_tL_R,
#         " ec_diag_tR_R = ",ec_diag_tR_R, " ec_diag_tR_L = ", ec_diag_tR_L)
#     end
#     return itR, itL, itIL, itIR
#     # , ec_diag_tL_L, ec_diag_tL_R, ec_diag_tR_R, ec_diag_tR_L
# end

# function HRHL(id_l, id_r, h_l, h_r, tL, tR, itR, itL, itIL, itIR, epsiPrec)
#     ptol =  min(1e-3, epsiPrec)/1000;
    
#     # generate combiners and combined indicies
#     cbL = combiner(id_l, id_l', tags="cbL");
#     idcbL = uniqueind(cbL, tL);
#     cbR = combiner(id_r, id_r', tags="cbR");
#     idcbR = uniqueind(cbR, tL);

#     ## calculate HL
#     # itML = -tL  + itR*itIL;
#     pjRI = replaceinds(itR,[id_r, id_r']=> [id_l, id_l']) * 
#     replaceinds(itIL, [id_l, id_l'] => [id_r, id_r']) / 
#     sqrt(
#         norm(
#             itR *replaceinds(itIL, [id_l, id_l'] => [id_r, id_r'])
#             ) 
#     );

#     itML = -tL + pjRI;
#     cbML = cbR * itML * cbL + diagITensor(ComplexF64,1, idcbR, idcbL);
#     # inds(cbML)

#     # itxL = h_l - replaceinds(h_l, [id_l, id_l'],[id_r, id_r'])*itR*itIL;
#     itxL = h_l - replaceinds(h_l * pjRI,[id_r, id_r']=> [id_l, id_l']);
#     cbxL = cbL *  itxL;
#     # inds(cbxL)
#     mtML = array(cbML);    mtxL = array(cbxL);
#     mtHL = bicgstabl(mtML, mtxL, 2, reltol = ptol);
#     rel_err_L_1 = norm(mtxL - mtML*mtHL);
#     itHL = ITensor(mtHL, idcbL);
#     rel_err_L_2 = norm(replaceinds(itHL *  cbML,[idcbR],[idcbL]) - cbxL);
#     itHL = itHL * dag(cbL);

    

#     ## calculate HR
#     # itMR = -tR +   itIR * itL;
#     pjIL = replaceinds(itIR,[id_r, id_r']=> [id_l, id_l'] )*
#     replaceinds(itL,[id_l, id_l'] => [id_r, id_r']) /
#     sqrt(
#         norm(
#             itIR * replaceinds(itL,[id_l, id_l'] => [id_r, id_r'])
#             )
#         );
#     itMR = -tR + pjIL;
#     cbMR = cbL* itMR * cbR  + diagITensor(ComplexF64,1, idcbL, idcbR);
#     # inds(cbMR)
#     # itxR = h_r - itIR * itL* replaceinds(h_r, [id_r, id_r'],[id_l, id_l']);
#     itxR = h_r - replaceinds(pjIL*  h_r,[id_l, id_l']=> [id_r, id_r']);
#     cbxR = cbR *  itxR;
#     # inds(cbxR)
#     mtMR = array(cbMR);    mtxR = array(cbxR);
#     mtHR = bicgstabl(mtMR, mtxR, 2, reltol = ptol);

#     rel_err_R_1 = norm(mtxR - mtMR*mtHR);
#     itHR = ITensor(mtHR, idcbR);
#     rel_err_R_2 = norm(replaceinds(itHR *  cbMR,[idcbL],[idcbR]) - cbxR);
#     itHR = itHR * dag(cbR);
#     if (round(rel_err_L_1 - rel_err_L_2, digits=6) == 0) & (round(rel_err_R_1 - rel_err_R_2, digits=6) == 0)
#         println("HRHL: no error")
#     else
#         println("HRHL: error in matching indicies")
#     end

#     return itHL, itHR

# end




function HRHL(id_l, id_r, h_l, h_r, tL, tR, itR, itL, itIL, itIR, epsiPrec)
    ptol =  min(1e-3, epsiPrec)/1000;
    
    # generate combiners and combined indicies
    cbL = combiner(id_l, id_l', tags="cbL");
    idcbL = uniqueind(cbL, tL);
    cbR = combiner(id_r, id_r', tags="cbR");
    idcbR = uniqueind(cbR, tL);

    ## calculate HL
    # itML = -tL  + itR*itIL;
    pjRI = replaceinds(itR,[id_r, id_r']=> [id_l, id_l']) * 
    replaceinds(itIL, [id_l, id_l'] => [id_r, id_r']);

    itML = -tL + pjRI;
    cbML = cbR * itML * cbL + diagITensor(ComplexF64,1, idcbR, idcbL);
    # inds(cbML)

    # itxL = h_l - replaceinds(h_l, [id_l, id_l'],[id_r, id_r'])*itR*itIL;
    itxL = h_l - replaceinds(h_l * pjRI,[id_r, id_r']=> [id_l, id_l']);
    cbxL = cbL *  itxL;
    # inds(cbxL)
    mtML = array(cbML);    mtxL = array(cbxL);
    mtHL = bicgstabl(mtML, mtxL, 2, reltol = ptol);
    rel_err_L_1 = norm(mtxL - mtML*mtHL);
    itHL = ITensor(mtHL, idcbL);
    rel_err_L_2 = norm(replaceinds(itHL *  cbML,[idcbR],[idcbL]) - cbxL);
    itHL = itHL * dag(cbL);

    

    ## calculate HR
    # itMR = -tR +   itIR * itL;
    pjIL = replaceinds(itIR,[id_r, id_r']=> [id_l, id_l'] )*
    replaceinds(itL,[id_l, id_l'] => [id_r, id_r']);
    itMR = -tR + pjIL;
    cbMR = cbL* itMR * cbR  + diagITensor(ComplexF64,1, idcbL, idcbR);
    # inds(cbMR)
    # itxR = h_r - itIR * itL* replaceinds(h_r, [id_r, id_r'],[id_l, id_l']);
    itxR = h_r - replaceinds(pjIL*  h_r,[id_l, id_l']=> [id_r, id_r']);
    cbxR = cbR *  itxR;
    # inds(cbxR)
    mtMR = array(cbMR);    mtxR = array(cbxR);
    mtHR = bicgstabl(mtMR, mtxR, 2, reltol = ptol);

    rel_err_R_1 = norm(mtxR - mtMR*mtHR);
    itHR = ITensor(mtHR, idcbR);
    rel_err_R_2 = norm(replaceinds(itHR *  cbMR,[idcbL],[idcbR]) - cbxR);
    itHR = itHR * dag(cbR);
    if (round(rel_err_L_1 - rel_err_L_2, digits=6) == 0) & (round(rel_err_R_1 - rel_err_R_2, digits=6) == 0)
        println("HRHL: no error")
    else
        println("HRHL: error in matching indicies")
    end

    return itHL, itHR

end



function calaCC(
    HaC, HC,
    id_l, id_d, id_r, 
    epsiPrec,
    shft = 1e2)

    ## calculation tolerance
    ptol =  min(1e-3, epsiPrec)/100;


    # combining indices 
    cbHaC_r = combiner(id_l, id_d,  id_r,  tags="cbHaC_r");
    cbHaC_l = combiner(id_l',id_d', id_r', tags="cbHaC_l");
    idcbHaC_r = uniqueind(cbHaC_r, HaC);
    # idcbHaC_l = uniqueind(cbHaC_l, HaC);

    cbHaC = cbHaC_l * HaC * cbHaC_r;

    cbHC_r = combiner(id_l,  id_r,  tags="cbHC_r");
    cbHC_l = combiner(id_l', id_r', tags="cbHC_l");
    idcbHC_r = uniqueind(cbHC_r, HC);
    # idcbHC_l = uniqueind(cbHC_l, HC);

    cbHC = cbHC_l * HC * cbHC_r;

    ## HaC eigen solve
    A = matrix(cbHaC) - shft * I;
    EaC, PaC =  powm(A,     inverse = false, shift = shft, tol = ptol, maxiter = 10000 );
    #           powm(A-σ*I, inverse = false, shift = σ,    tol = 1e-4, maxiter = 200);

    # PaC   = PaC * norm(PaC[1])/PaC[1];
    PaC = PaC/norm(PaC); # normalize size and phase
    # psR = psR * norm(psR[1])/psR[1]; psR = psR/norm(psR); # normalize size and phase

    itPaC = ITensor(ComplexF64, PaC, idcbHaC_r);
    # norm(array(replaceinds( (cbHaC * itPaC ) ,[idcbHaC_l] ,[idcbHaC_r]) - EaC * itPaC))
    itPaC = cbHaC_r * itPaC;
    # inds(itPaC)
    ec_aC =  norm(array(replaceinds( HaC * itPaC , [id_l', id_d', id_r'],  [id_l, id_d, id_r])- EaC * itPaC));

    ## HC eigen solve
    A = matrix(cbHC) - shft * I;
    EC, PC =  powm(A,     inverse = false, shift = shft, tol = ptol, maxiter = 10000 );
    #         powm(A-σ*I, inverse = false, shift = σ,    tol = 1e-4, maxiter = 200);


    # PC    = PC * norm(PC[1])/PC[1]; 
    PC = PC/norm(PC); # normalize size and phase


    itPC = ITensor(ComplexF64, PC, idcbHC_r);
    # norm(array(replaceinds( (cbHC * itPC ) ,[idcbHC_l] ,[idcbHC_r]) - EC * itPC))

    itPC = cbHC_r * itPC;
    ec_C =  norm(array(replaceinds( HC * itPC , [id_l', id_r'],  [id_l, id_r])- EC * itPC));


    return itPaC, itPC, EaC,  EC,  ec_aC,  ec_C
    #      aC,    C   , EaC,  EC,  ϵ_aC,   ϵ_C



end





