using ITensors
using Dates
using LinearAlgebra


#######################################################################
######################## two-site Hamiltonians ########################
#######################################################################
function hTFI(h0, d, id_u, id_d, id_u1, id_d1, sn=3)
    # temporal indices

    # Pauli matrices
    σ_x = ITensor(id_u, id_d); σ_y = ITensor(id_u, id_d); σ_z = ITensor(id_u, id_d);
    σ_0 = ITensor(id_u, id_d);

    # σ_x[1,2] = 1; σ_x[2,1]= 1; σ_y[1,2] = -1im; σ_y[2,1] = 1im;  σ_z[1,1] = 1; σ_z[2,2]= -1; σ_0[1,1] = 1; σ_0[2,2]= 1;
    σ_x[1,2] = 1.; σ_x[2,1]= 1.; σ_y[1,2] = -1im; σ_y[2,1] = 1im;  σ_z[1,1] = 1.; σ_z[2,2]= -1.; σ_0[1,1] = 1.; σ_0[2,2]= 1.;

    # ingredients
    xx = replaceinds(σ_x, [id_u, id_d], [id_u1[sn], id_d1[sn]] ) * replaceinds(σ_x, [id_u, id_d], [id_u1[sn+1], id_d1[sn+1]] );
    yy = replaceinds(σ_y, [id_u, id_d], [id_u1[sn], id_d1[sn]] ) * replaceinds(σ_y, [id_u, id_d], [id_u1[sn+1], id_d1[sn+1]] );
    zz = replaceinds(σ_z, [id_u, id_d], [id_u1[sn], id_d1[sn]] ) * replaceinds(σ_z, [id_u, id_d], [id_u1[sn+1], id_d1[sn+1]] );

    h_TFI = - xx - (h0/2.) * replaceinds(σ_z, [id_u, id_d], [id_u1[sn], id_d1[sn]] ) * replaceinds(σ_0, [id_u, id_d], [id_u1[sn+1], id_d1[sn+1]] ) -
    (h0/2.)* replaceinds(σ_0, [id_u, id_d], [id_u1[sn], id_d1[sn]] ) * replaceinds(σ_z, [id_u, id_d], [id_u1[sn+1], id_d1[sn+1]] );

    return h_TFI
end


function hXXZ(Δ0, d, id_u, id_d, id_u1, id_d1, sn = 3)
    # temporal indices
    in_ud = Index(d,"index_ud");
    
    # Pauli matrices
    σ_x = ITensor(id_u, id_d); σ_y = ITensor(id_u, id_d); σ_z = ITensor(id_u, id_d);
    σ_0 = ITensor(id_u, id_d);

    σ_x[1,2] = 1; σ_x[2,1]= 1; σ_y[1,2] = -1im; σ_y[2,1] = 1im;  σ_z[1,1] = 1; σ_z[2,2]= -1; σ_0[1,1] = 1; σ_0[2,2]= 1;

    # ingredients
    xx = replaceinds(σ_x, [id_u, id_d], [id_u1[sn], id_d1[sn]] ) * replaceinds(σ_x, [id_u, id_d], [id_u1[sn+1], id_d1[sn+1]] );
    yy = replaceinds(σ_y, [id_u, id_d], [id_u1[sn], id_d1[sn]] ) * replaceinds(σ_y, [id_u, id_d], [id_u1[sn+1], id_d1[sn+1]] );
    zz = replaceinds(σ_z, [id_u, id_d], [id_u1[sn], id_d1[sn]] ) * replaceinds(σ_z, [id_u, id_d], [id_u1[sn+1], id_d1[sn+1]] );

    h_XXZ = xx + yy + Δ0*zz;
    return h_XXZ 

end




#####################################################################
######################## two-site h_l & h_r  ########################
#####################################################################
function TwohLhR(h_loc, a_l, a_r, id_d, id_l, id_r, id_lr, id_u1, id_d1, sn=3)
    # inputs
    # h_loc : two-site local Hamiltonians
    # a_l, a_r : MPSs
    # id_l, id_r, id_lr : virtual indices
    # id_d, id_r, id_u1, id_d1 : physical indices
    
    
    # h_l and h_r construction 
    a_l_uu = replaceinds(a_l,[id_l, id_d, id_r], [id_lr', id_u1[sn], id_lr]) * replaceinds(a_l,[id_d, id_l, id_r], [id_u1[sn+1], id_lr, id_l]);
    a_l_dd = replaceinds(conj(a_l_uu), [id_u1[sn], id_u1[sn+1], id_l], [id_d1[sn], id_d1[sn+1], id_l']);
    # a_l_ud = a_l_uu * a_l_dd;
    h_l = a_l_uu * h_loc *  a_l_dd;



    a_r_uu = replaceinds(a_r,[id_l, id_d, id_r], [id_r, id_u1[sn], id_lr]) * replaceinds(a_r,[id_l, id_d, id_r], [id_lr, id_u1[sn+1], id_lr']);
    a_r_dd = replaceinds(conj(a_r_uu), [id_r, id_u1[sn], id_u1[sn+1]], [id_r', id_d1[sn], id_d1[sn+1]]);
    # a_r_ud = a_r_uu * a_r_dd;
    h_r = a_r_uu * h_loc *a_r_dd;

    return h_l, h_r

end





# function effH_1(h_loc,
#     itHL, itHR, 
#     aL, aR, 
#     id_l, id_r, id_d, id_u1, id_d1, 
#     id_lr, id_ud, 
#     sn = 3)

#     HaC_1 = replaceinds(aL,[id_l, id_d, id_r ] =>[id_lr, id_ud, id_l]) *
#         replaceinds(h_loc,[id_u1[sn], id_u1[sn+1], id_d1[sn], id_d1[sn+1]]=> [id_ud, id_d, id_ud', id_d']) *
#         replaceinds(conj(aL), [id_l, id_d, id_r] => [id_lr, id_ud', id_l']) * diagITensor(ComplexF64, 1, [id_r, id_r']);

#     HaC_2 = diagITensor(ComplexF64, 1, [id_l, id_l']) *
#             replaceinds(aR, [id_l, id_d, id_r] => [id_r, id_ud, id_lr]) *
#             replaceinds(h_loc,[id_u1[sn], id_u1[sn+1], id_d1[sn], id_d1[sn+1]]=> [id_d, id_ud, id_d', id_ud']) *
#             replaceinds(conj(aR), [id_l, id_d, id_r] => [id_r', id_ud', id_lr]);

#     HaC_3 = itHL * diagITensor(ComplexF64, 1, [id_d, id_d']) * diagITensor(ComplexF64, 1, [id_r, id_r']);
#     HaC_4 = diagITensor(ComplexF64, 1, [id_l, id_l']) * diagITensor(ComplexF64, 1, [id_d, id_d']) * itHR;

#     ## calculate HaC
#     HaC = HaC_1 + HaC_2 + HaC_3 + HaC_4;
#     # inds(HaC)

#     ## calculate HC_1 ~ HC_3
#     HC_1 = replaceinds(aL, [id_l, id_d, id_r] => [id_lr, id_ud, id_l]) *
#             replaceinds(aR, [id_l, id_d, id_r] => [id_r, id_ud', id_lr']) *
#             replaceinds(h_loc, [id_u1[sn], id_u1[sn+1], id_d1[sn], id_d1[sn+1]] => [id_ud, id_ud', id_ud'', id_ud''']) *
#             replaceinds(conj(aL), [id_l, id_d, id_r] => [id_lr, id_ud'', id_l']) *
#             replaceinds(conj(aR), [id_l, id_d, id_r] => [id_r', id_ud''', id_lr']);
#     # inds(HC_1)
#     HC_2 = itHL * diagITensor(ComplexF64, 1, id_r, id_r');
#     HC_3 = diagITensor(ComplexF64, 1, id_l, id_l') * itHR;

#     HC = HC_1 + HC_2 + HC_3;


#     return HaC, HC
# end


function effH_1(h_loc,
    itHL, itHR, 
    aL, aR, 
    id_l, id_r, id_d, id_u1, id_d1, 
    id_lr, id_ud, 
    sn = 3)


    ## calculate HC_1 ~ HC_3
    HC_1 = replaceinds(aL, [id_l, id_d, id_r] => [id_lr, id_ud, id_l]) *
    replaceinds(aR, [id_l, id_d, id_r] => [id_r, id_ud', id_lr']) *
    replaceinds(h_loc, [id_u1[sn], id_u1[sn+1], id_d1[sn], id_d1[sn+1]] => [id_ud, id_ud', id_ud'', id_ud''']) *
    replaceinds(conj(aL), [id_l, id_d, id_r] => [id_lr, id_ud'', id_l']) *
    replaceinds(conj(aR), [id_l, id_d, id_r] => [id_r', id_ud''', id_lr']);
    # inds(HC_1)
    HC_2 = itHL * diagITensor(ComplexF64, 1, id_r, id_r');
    HC_3 = diagITensor(ComplexF64, 1, id_l, id_l') * itHR;

    HC = HC_1 + HC_2 + HC_3;

    HaC_1 = replaceinds(aL,[id_l, id_d, id_r ] =>[id_lr, id_ud, id_l]) *
        replaceinds(h_loc,[id_u1[sn], id_u1[sn+1], id_d1[sn], id_d1[sn+1]]=> [id_ud, id_d, id_ud', id_d']) *
        replaceinds(conj(aL), [id_l, id_d, id_r] => [id_lr, id_ud', id_l']) * diagITensor(ComplexF64, 1, [id_r, id_r']);

    HaC_2 = diagITensor(ComplexF64, 1, [id_l, id_l']) *
            replaceinds(aR, [id_l, id_d, id_r] => [id_r, id_ud, id_lr]) *
            replaceinds(h_loc,[id_u1[sn], id_u1[sn+1], id_d1[sn], id_d1[sn+1]]=> [id_d, id_ud, id_d', id_ud']) *
            replaceinds(conj(aR), [id_l, id_d, id_r] => [id_r', id_ud', id_lr]);

    # HaC_3 = HC_2 * diagITensor(ComplexF64, 1, [id_d, id_d']);
    # HaC_4 = HC_3 * diagITensor(ComplexF64, 1, [id_d, id_d']);

    ## calculate HaC
    HaC = HaC_1 + HaC_2 + 
    HC_2 * diagITensor(ComplexF64, 1, [id_d, id_d']) + 
    HC_3 * diagITensor(ComplexF64, 1, [id_d, id_d']);
    # inds(HaC)




    return HaC, HC
end


