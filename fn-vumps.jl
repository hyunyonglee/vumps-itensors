function vumps_u1(
    id_l, id_r, id_d, id_u1, id_d1, 
    id_lr, id_ud, 
    aC, C, aL, aR, 
    h_loc,
    epsiPrec, sn,
    n_V, B_cut,
    HLHR_check = false
    )

    # empty tuples for the storing errors
    s_epsiPrec = []; s_epsiL =[]; s_epsiR = []; s_ec_aC = []; s_ec_C = [];
    s_Babs = []; 

    # empty tuples for the storing energies
    s_EaC = [];  s_EC = [];


    t0 = now();

    for i_V in 1:n_V
        println("##############################################################################################")
        println("i_V = ", i_V);

        ## left and right transfer matrices
        tL = aL*setprime(conj(aL),1,[id_r, id_l] );
        tR = aR*setprime(conj(aR),1,[id_r, id_l] );

        t3=now();
        ## Contracted local Hamiltonian
        h_l, h_r  = TwohLhR(h_loc, aL,  aR,  id_d, id_l, id_r, id_lr, id_u1, id_d1, sn);
        #           TwohLhR(h_loc, a_l, a_r, id_d, id_l, id_r, id_lr, id_u1, id_d1, sn=3)
        t4=now(); Δt_hlhr = (t4 -t3)/ Millisecond(1);


        ## Eigenvectors of transfer matrices
        itR, itL, itIL, itIR = calLR(id_l, id_r, tL, tR, epsiPrec, -1);
        #                      calLR(id_l, id_r, tL, tR, epsiPrec, shft = -1)
        t3=now(); Δt_LR = (t3 -t4)/ Millisecond(1);

        ## Calculate HL and HR
        itHL, itHR = HRHL(id_l, id_r, h_l, h_r, tL, tR, itR, itL, itIL, itIR, epsiPrec);
        #            HRHL(id_l, id_r, h_l, h_r, tL, tR, itR, itL, itIL, itIR, epsiPrec)
        # norm(array(fitHL - itHL))
        t4=now(); Δt_HLHR = (t4 -t3)/ Millisecond(1);


        ################ for HL HR convergence test ################
        ############################################################
        if HLHR_check == true
        
            pjRI = replaceinds(itR,[id_r, id_r']=> [id_l, id_l']) * 
            replaceinds(itIL, [id_l, id_l'] => [id_r, id_r']);
            pjIL = replaceinds(itIR,[id_r, id_r']=> [id_l, id_l'] )*
                replaceinds(itL,[id_l, id_l'] => [id_r, id_r']);
            h_l_t = h_l - replaceinds(h_l * pjRI, [id_r, id_r'], [id_l, id_l']);
            h_r_t = h_r - replaceinds(h_r * pjIL, [id_l, id_l'], [id_r, id_r']);
            er_hl_15 = norm(itHL 
            - replaceinds(itHL,[id_l, id_l'] => [id_r, id_r']) *
            replaceinds(tL,[id_l, id_l', id_r, id_r'], [id_r, id_r', id_l, id_l']) +
            replaceinds(itHL,[id_l, id_l'] => [id_r, id_r']) *
            replaceinds(pjRI,[id_l, id_l', id_r, id_r'], [id_r, id_r', id_l, id_l']) -
            h_l_t);
            er_hr_15 = norm(itHR 
            - replaceinds(itHR,[id_r, id_r'] => [id_l, id_l']) *
            replaceinds(tR, [id_r, id_r', id_l, id_l'],[id_l, id_l', id_r, id_r']) +
            replaceinds(itHR, [id_r, id_r'] => [id_l, id_l']) *
            replaceinds(pjIL, [id_r, id_r', id_l, id_l'], [id_l, id_l', id_r, id_r']) -
            h_r_t);

            er_hl_14 = norm(itHL 
            - replaceinds(itHL,[id_l, id_l'] => [id_r, id_r']) *
            replaceinds(tL,[id_l, id_l', id_r, id_r'], [id_r, id_r', id_l, id_l']) -
            h_l_t);
            er_hr_14 = norm(itHR 
            - replaceinds(itHR,[id_r, id_r'] => [id_l, id_l']) *
            replaceinds(tR, [id_r, id_r', id_l, id_l'],[id_l, id_l', id_r, id_r']) -
            h_r_t);
            println(er_hl_14,"\t",er_hr_14,"\n")
            println(er_hl_15,"\t",er_hr_15,"\n")
        
        end
        ############################################################
        ############################################################

        ## Calculate Effective Hamiltonians; fHaC and fHC
        HaC, HC = effH_1(h_loc, itHL, itHR, aL, aR, id_l, id_r, id_d, id_u1, id_d1, id_lr, id_ud, sn);
        #         effH_1_2(h_loc, itHL, itHR, aL, aR, id_l, id_r, id_d, id_u1, id_d1, id_lr, id_ud, sn =
        t3=now(); Δt_eff = (t3 -t4)/ Millisecond(1);

        # Calculate aC and C for next step. Calculate energies. Keep errors.
        aC, C,  EaC,  EC,  ec_aC, ec_C = calaCC(HaC, HC, id_l, id_d, id_r, epsiPrec);
        # calaCC(HaC, HC, id_l, id_d, id_r, epsiPrec, shft = 1e2)
        append!(s_EaC,[EaC]); append!(s_EC, [EC]); append!(s_ec_aC,[ec_aC]); append!(s_ec_C,[ec_C]); 
        t4=now(); Δt_eigen = (t4 -t3)/ Millisecond(1);

        ## left and right canonical MPSs
        aL, aR, epsiPrec, epsiL, epsiR = aLRsvd(aC, C, id_l, id_r, id_d);
        t3=now(); Δt_aLaR = (t3 -t4)/ Millisecond(1);
        append!(s_epsiL,[epsiL]); append!(s_epsiR,[epsiR]); append!(s_epsiPrec, [epsiPrec]);
        
        # BL = aC - replaceinds(aL,[id_r], [id_lr]) * replaceinds(C, [id_l], [id_lr]) ;
        # BR = aC - replaceinds(C,[id_r], [id_lr])  * replaceinds(aR, [id_l], [id_lr]) ;
        # absB = max( norm(BL),norm(BR));
        absB = epsiPrec;
        append!(s_Babs, [absB]);


        t1 = now(); 
        Δt_tot = t1-t0;
        t0 = now();
        println("Δt_hlhr = ", Δt_hlhr,"\n","Δt_LR = ", Δt_LR, "\n", "Δt_HLHR = ", Δt_HLHR, "\n",
        "Δt_eff = ", Δt_eff, "\n", "Δt_eigen = ", Δt_eigen, "\n",
        "Δt_aLaR = ", Δt_aLaR, "\n", "Δt_tot = ", Δt_tot, "\t ", "epsiPrec = ", epsiPrec)

        if absB < B_cut
            break;
        end
    end
    
    return aC, C, aL, aR, s_epsiPrec, s_epsiL, s_epsiR, s_ec_aC, s_ec_C, s_Babs, s_EaC, s_EC


end

function vumps_u1_t1(
    id_l, id_r, id_d, id_u1, id_d1, 
    id_lr, id_ud, 
    aC, C, aL, aR, 
    h_loc,
    epsiPrec, sn,
    n_V, B_cut,
    HLHR_check = false
    )

    # empty tuples for the storing errors
    s_epsiPrec = []; s_epsiL =[]; s_epsiR = []; s_ec_aC = []; s_ec_C = [];
    s_Babs = []; 

    # empty tuples for the storing energies
    s_EaC = [];  s_EC = [];


    t0 = now();

    for i_V in 1:n_V
        println("##############################################################################################")
        println("i_V = ", i_V);

        ## left and right transfer matrices
        tL = aL*setprime(conj(aL),1,[id_r, id_l] );
        tR = aR*setprime(conj(aR),1,[id_r, id_l] );

        t3=now();
        ## Contracted local Hamiltonian
        h_l, h_r  = TwohLhR(h_loc, aL,  aR,  id_d, id_l, id_r, id_lr, id_u1, id_d1, sn);
        #           TwohLhR(h_loc, a_l, a_r, id_d, id_l, id_r, id_lr, id_u1, id_d1, sn=3)
        t4=now(); Δt_hlhr = (t4 -t3)/ Millisecond(1);


        ## Eigenvectors of transfer matrices
        # itR, itL, itIL, itIR = calLR(id_l, id_r, tL, tR, epsiPrec, -1);
        #                      calLR(id_l, id_r, tL, tR, epsiPrec, shft = -1)
        itL = C * setprime(conj(C), 1, id_r);
        itIR = diagITensor(ComplexF64,1, id_r, id_r');
        itR = C * setprime(conj(C), 1, id_l);
        itIL = diagITensor(ComplexF64,1, id_l, id_l');
        t3=now(); Δt_LR = (t3 -t4)/ Millisecond(1);

        ## Calculate HL and HR
        itHL, itHR = HRHL(id_l, id_r, h_l, h_r, tL, tR, itR, itL, itIL, itIR, epsiPrec);
        #            HRHL(id_l, id_r, h_l, h_r, tL, tR, itR, itL, itIL, itIR, epsiPrec)
        # norm(array(fitHL - itHL))
        t4=now(); Δt_HLHR = (t4 -t3)/ Millisecond(1);


        ################ for HL HR convergence test ################
        ############################################################
        if HLHR_check == true
        
            pjRI = replaceinds(itR,[id_r, id_r']=> [id_l, id_l']) * 
            replaceinds(itIL, [id_l, id_l'] => [id_r, id_r']);
            pjIL = replaceinds(itIR,[id_r, id_r']=> [id_l, id_l'] )*
                replaceinds(itL,[id_l, id_l'] => [id_r, id_r']);
            h_l_t = h_l - replaceinds(h_l * pjRI, [id_r, id_r'], [id_l, id_l']);
            h_r_t = h_r - replaceinds(h_r * pjIL, [id_l, id_l'], [id_r, id_r']);
            er_hl_15 = norm(itHL 
            - replaceinds(itHL,[id_l, id_l'] => [id_r, id_r']) *
            replaceinds(tL,[id_l, id_l', id_r, id_r'], [id_r, id_r', id_l, id_l']) +
            replaceinds(itHL,[id_l, id_l'] => [id_r, id_r']) *
            replaceinds(pjRI,[id_l, id_l', id_r, id_r'], [id_r, id_r', id_l, id_l']) -
            h_l_t);
            er_hr_15 = norm(itHR 
            - replaceinds(itHR,[id_r, id_r'] => [id_l, id_l']) *
            replaceinds(tR, [id_r, id_r', id_l, id_l'],[id_l, id_l', id_r, id_r']) +
            replaceinds(itHR, [id_r, id_r'] => [id_l, id_l']) *
            replaceinds(pjIL, [id_r, id_r', id_l, id_l'], [id_l, id_l', id_r, id_r']) -
            h_r_t);

            er_hl_14 = norm(itHL 
            - replaceinds(itHL,[id_l, id_l'] => [id_r, id_r']) *
            replaceinds(tL,[id_l, id_l', id_r, id_r'], [id_r, id_r', id_l, id_l']) -
            h_l_t);
            er_hr_14 = norm(itHR 
            - replaceinds(itHR,[id_r, id_r'] => [id_l, id_l']) *
            replaceinds(tR, [id_r, id_r', id_l, id_l'],[id_l, id_l', id_r, id_r']) -
            h_r_t);
            println(er_hl_14,"\t",er_hr_14,"\n")
            println(er_hl_15,"\t",er_hr_15,"\n")
        
        end
        ############################################################
        ############################################################

        ## Calculate Effective Hamiltonians; fHaC and fHC
        HaC, HC = effH_1(h_loc, itHL, itHR, aL, aR, id_l, id_r, id_d, id_u1, id_d1, id_lr, id_ud, sn);
        #         effH_1_2(h_loc, itHL, itHR, aL, aR, id_l, id_r, id_d, id_u1, id_d1, id_lr, id_ud, sn =
        t3=now(); Δt_eff = (t3 -t4)/ Millisecond(1);

        # Calculate aC and C for next step. Calculate energies. Keep errors.
        aC, C,  EaC,  EC,  ec_aC, ec_C = calaCC(HaC, HC, id_l, id_d, id_r, epsiPrec);
        # calaCC(HaC, HC, id_l, id_d, id_r, epsiPrec, shft = 1e2)
        append!(s_EaC,[EaC]); append!(s_EC, [EC]); append!(s_ec_aC,[ec_aC]); append!(s_ec_C,[ec_C]); 
        t4=now(); Δt_eigen = (t4 -t3)/ Millisecond(1);

        ## left and right canonical MPSs
        aL, aR, epsiPrec, epsiL, epsiR = aLRsvd(aC, C, id_l, id_r, id_d);
        t3=now(); Δt_aLaR = (t3 -t4)/ Millisecond(1);
        append!(s_epsiL,[epsiL]); append!(s_epsiR,[epsiR]); append!(s_epsiPrec, [epsiPrec]);
        
        # BL = aC - replaceinds(aL,[id_r], [id_lr]) * replaceinds(C, [id_l], [id_lr]) ;
        # BR = aC - replaceinds(C,[id_r], [id_lr])  * replaceinds(aR, [id_l], [id_lr]) ;
        # absB = max( norm(BL),norm(BR));
        absB = epsiPrec;
        append!(s_Babs, [absB]);


        t1 = now(); 
        Δt_tot = t1-t0;
        t0 = now();
        println("Δt_hlhr = ", Δt_hlhr,"\n","Δt_LR = ", Δt_LR, "\n", "Δt_HLHR = ", Δt_HLHR, "\n",
        "Δt_eff = ", Δt_eff, "\n", "Δt_eigen = ", Δt_eigen, "\n",
        "Δt_aLaR = ", Δt_aLaR, "\n", "Δt_tot = ", Δt_tot, "\t ", "epsiPrec = ", epsiPrec)

        if absB < B_cut
            break;
        end
    end
    
    return aC, C, aL, aR, s_epsiPrec, s_epsiL, s_epsiR, s_ec_aC, s_ec_C, s_Babs, s_EaC, s_EC


end