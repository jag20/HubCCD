      Subroutine GetG1(G1a,G1b,T1a,T1b,T2aa,T2ab,T2bb,FockA,FockB,ERIaa,ERIab,ERIbb,oA,oB,NBF)
      Integer,        Intent(In)  :: oA, oB, NBF
      double precision, Intent(In)  :: FockA(NBF,NBF), FockB(NBF,NBF) 
      double precision, Intent(In)  :: ERIaa(NBF,NBF,NBF,NBF),&
                                       ERIab(NBF,NBF,NBF,NBF),&
                                       ERIbb(NBF,NBF,NBF,NBF)
      double precision, Intent(In)  :: T2aa(1:oA,1:oA,oa+1:NBF,oa+1:NBF)
      double precision, Intent(In)  :: T2ab(1:oA,1:oB,oa+1:NBF,ob+1:NBF)
      double precision, Intent(In)  :: T2bb(1:ob,1:oB,ob+1:NBF,ob+1:NBF)
      double precision, Intent(In)  :: &
                                       T1a(1:oA,oa+1:NBF),&
                                       T1b(1:oB,ob+1:NBF)
      double precision, Intent(Out) :: &
                                     G1a(1:oA,oa+1:NBF),&
                                     G1b(1:oB,ob+1:NBF)
      !f2py depend(NBF) FockA, FockB, EriAA, EriAB, ERIbb, G1a, G1b, T2aa, T2ab,T2bb, T1a, T1b, G1a, G1ab
      !f2py depend(oA) T2aa, T2ab, T1a, G1a
      !f2py depend(ob) T2ab, T2bb, T1b, G1b
! Local variables
      Integer :: I, J, A, B, K, L, C, D, vA, vB
      double precision :: Zero =0.0e0, F12 = 0.50e0, F14 = 0.25e0
      !New intermediate terms for CCSD   
      Double Precision, allocatable :: Jca(:,:), Jbca(:,:), Jik(:,:), Jbik(:,:),&
                                  Wikac(:,:,:,:), Wbikac(:,:,:,:),&
                                  Xikac(:,:,:,:), Xbikac(:,:,:,:),&
                                  Fik(:,:), Fbik(:,:), Ica(:,:), Ibca(:,:),&
                                  Iik(:,:), Ick(:,:), Ibik(:,:), Ibck(:,:)
                                  
         vA = oa+1
         vB = ob+1


!===============================================================================!
! This routine gets the right hand side of the singles equations for CCSD.      !
! Terms will be grouped on my notes, based on the equations given    !
! c.a. p. 308 in Shavitt and Bartlett                                           ! 
!===============================================================================!

      Allocate(Jca(Va:NBF,Va:NBF),               &
               Jbca(Vb:NBF,Vb:NBF),              &
               Jik(1:Va,1:Va),               &
               Jbik(1:Vb,1:Vb),              &
               Wikac(1:oA,1:oA,vA:NBF,vA:NBF),   &
               Wbikac(1:oB,1:oB,vB:NBF,vB:NBF),  &
               Xikac(1:oA,1:oB,vA:NBF,vB:NBF),   &
               Xbikac(1:oB,1:oA,vB:NBF,vA:NBF),  &
               Fik(1:oA,1:oA),               &
               Fbik(1:oB,1:oB),              &
               Ica(vA:NBF,va:NBF),               & 
               Ibca(vB:NBF,vB:NBF),              & 
               Iik(1:oA,1:oA),               & 
               Ibik(1:oB,1:oB),              & 
               Ick(vA:NBF,1:oA),               & 
               Ibck(vB:NBF,1:oB))
!Get Fock
        G1a = FockA(1:oA,vA:NBF)
        G1b = FockB(1:oB,vB:NBF)

!Line 2 from Notes 
! alpha
        Do I = 1, oA
        Do A = vA, NBF
                G1a(I,A) = G1a(I,A) + Sum(FockA(1:oA,vA:NBF)*T2aa(I,:,A,:)) 
                G1a(I,A) = G1a(I,A) + Sum(FockB(1:oB,vB:NBF)*T2ab(I,:,A,:)) 
        End Do
        End Do
! beta
        Do I = 1, oB
        Do A = vB, NBF
                G1b(I,A) = G1b(I,A) + Sum(FockB(1:oB,vB:NBF)*T2bb(I,:,A,:)) 
                G1b(I,A) = G1b(I,A) + Sum(FockA(1:oA,vA:NBF)*T2ab(:,I,:,A)) 
        End Do
        End Do

!Line 3 and 4 from Notes
! alpha
        Do I = 1, oA
        Do A = vA, NBF
              Do K = 1, oA
                G1a(I,A) = G1a(I,A) + F12*Sum(ERIaa(vA:NBF,vA:nbf,A,K)*T2aa(I,K,:,:))
                Do L = 1, oA
                Do C = vA, NBF
                  G1a(I,A) = G1a(I,A) - F12*ERIaa(I,C,K,L)*T2aa(K,L,A,C)
                End Do
                End Do
              End Do
              Do K = 1, oB
                G1a(I,A) = G1a(I,A) + Sum(ERIab(vA:NBF,vB:NBF,A,K)*T2ab(I,K,:,:))
              End Do
              Do K = 1, oA
                Do L = 1, oB
                Do C = vB, NBF
                  G1a(I,A) = G1a(I,A) - ERIab(I,C,K,L)*T2ab(K,L,A,C)
                End Do
                End Do
              End Do
        End Do
        End Do
! beta
        Do I = 1, oB
        Do A = vB, NBF
              Do K = 1, oB
                 G1b(I,A) = G1b(I,A) + F12*Sum(ERIbb(vB:NBF,vB:nbf,A,K)*T2bb(I,K,:,:))
                Do L = 1, oB
                Do C = vB, NBF
                  G1b(I,A) = G1b(I,A) - F12*ERIbb(I,C,K,L)*T2bb(K,L,A,C)
                End Do
                End Do
              End Do
              Do K = 1, oA
                G1b(I,A) = G1b(I,A) +  Sum(ERIab(vA:NBF,vB:NBF,K,A)*T2ab(K,I,:,:))
              End Do
              Do K = 1, oB
                Do L = 1, oA
                Do C = vA, NBF
                  G1b(I,A) = G1b(I,A) - ERIab(C,I,L,K)*T2ab(L,K,C,A)
                End Do
                End Do
              End Do
        End Do
        End Do
              
!Line 5 from Notes
! alpha
        Do I = 1, oA
        Do A = vA, NBF
                G1a(I,A) = G1a(I,A) + Sum(ERIaa(I,vA:NBF,A,1:oA)*Transpose(T1a(:,:)))
                G1a(I,A) = G1a(I,A) + Sum(ERIab(I,vB:NBF,A,1:oB)*Transpose(T1b(:,:)))
        End Do
        End Do
! beta
        Do I = 1, oB
        Do A = vB, NBF
                G1b(I,A) = G1b(I,A) + Sum(ERIbb(I,vB:NBF,A,1:oB)*Transpose(T1b(:,:)))
                G1b(I,A) = G1b(I,A) + Sum(ERIab(vA:NBF,I,1:oA,A)*Transpose(T1a(:,:)))
        End Do
        End Do
       ! write(*,*) "T1a = ", G1a
       ! write(*,*) "T1b = ", G1b


!Construct Jca 
        Jca = Zero
        Do C = vA, NBF
        Do A = vA, NBF
                Do D = vA, NBF
                  Jca(C,A) = Jca(C,A) + F12*Sum(ERIaa(C,D,1:oA,1:oA)*T2aa(:,:,A,D))
                End Do
                Do D = vB, NBF
                  Jca(C,A) = Jca(C,A) + Sum(ERIab(C,D,1:oA,1:oB)*T2ab(:,:,A,D))
                End Do
        End Do
        End Do
        Jca = -Jca
!Construct Jbca 
        Jbca = Zero
        Do C = vB, NBF
        Do A = vB, NBF
                Do D = vB, NBF
                  Jbca(C,A) = Jbca(C,A) + F12*Sum(ERIbb(C,D,1:oB,1:oB)*T2bb(:,:,A,D))
                End Do
                Do D = vA, NBF
                  Jbca(C,A) = Jbca(C,A) +  Sum(ERIab(D,C,1:oA,1:oB)*T2ab(:,:,D,A))
                End Do
        End Do
        End Do
        Jbca = -Jbca
!Use Jca
        Do I = 1, oA
        Do A = vA, NBF
                G1a(I,A) = G1a(I,A) + Sum(Jca(:,A)*T1a(I,:))

        End Do
        End Do
!Use Jbca
        Do I = 1, oB
        Do A = vB, NBF
                G1b(I,A) = G1b(I,A) + Sum(Jbca(:,A)*T1b(I,:))

        End Do
        End Do

!Construct Jik
        Jik = Zero
        Do I = 1, oA
        Do K = 1, oA
          Do L = 1, oA
          Jik(I,K) = Jik(I,K) + F12*Sum(ERIaa(vA:NBF,vA:NBF,K,L)*T2aa(I,L,:,:))
          End Do 
          Do L = 1, oB
          Jik(I,K) = Jik(I,K) + Sum(ERIab(vA:NBF,vB:NBF,K,L)*T2ab(I,L,:,:))
          End Do 
        End Do 
        End Do 
        Jik = -Jik 
!Construct Jbik
        Jbik = Zero
        Do I = 1, oB
        Do K = 1, oB
          Do L = 1, oB
          Jbik(I,K) = Jbik(I,K) + F12*Sum(ERIbb(vB:NBF,vB:NBF,K,L)*T2bb(I,L,:,:))
          End Do 
          Do L = 1, oA
          Jbik(I,K) = Jbik(I,K) + Sum(ERIab(vA:NBF,vB:NBF,L,K)*T2ab(L,I,:,:))
          End Do 
        End Do 
        End Do 
        Jbik = -Jbik 
!Use Jik
        Do I = 1, oA
        Do A = vA,NBF
          G1a(I,A) = G1a(I,A) + Sum(Jik(I,:)*T1a(:,A))
        End Do
        End Do
!Use Jbik
        Do I = 1, oB
        Do A = vB,NBF
          G1b(I,A) = G1b(I,A) + Sum(Jbik(I,:)*T1b(:,A))
        End Do
        End Do


!Construct Wikac
        Wikac = Zero
        Do I = 1, oA
        Do A = vA, NBF
          Do K = 1, oA
          Do C = vA, NBF
          Wikac(I,K,A,C) = Wikac(I,K,A,C) + Sum(ERIaa(C,vA:NBF,K,1:oA)*Transpose(T2aa(:,I,:,A)))&
                                          + Sum(ERIab(C,vB:NBF,K,1:oB)*Transpose(T2ab(I,:,A,:)))
          End Do
          End Do
        End Do
        End Do
!Construct Wbikac
        Wbikac = Zero
        Do I = 1, oB
        Do A = vB, NBF
          Do K = 1, oB
          Do C = vB, NBF
          Wbikac(I,K,A,C) = Wbikac(I,K,A,C) + Sum(ERIbb(C,vB:NBF,K,1:oB)*Transpose(T2bb(:,I,:,A)))&
                                            + Sum(ERIab(vA:NBF,C,1:oA,K)*Transpose(T2ab(:,I,:,A)))
          End Do
          End Do
        End Do
        End Do
!Use Wikac
        Do I = 1, oA
        Do A = vA, NBF
          G1a(I,A) = G1a(I,A) + Sum(Wikac(I,:,A,:)*(T1a(:,:)))
        End Do
        End Do
!Use Wbikac
        Do I = 1, oB
        Do A = vB, NBF
          G1b(I,A) = G1b(I,A) + Sum(Wbikac(I,:,A,:)*(T1b(:,:)))
        End Do
        End Do


!Construct Xikac
        Xikac = Zero
        Do I = 1, oA
        Do A = vA, NBF
          Do K = 1, oB
          Do C = vB, NBF
            Xikac(I,K,A,C) = Xikac(I,K,A,C) + Sum(ERIab(vA:NBF,C,1:oA,K)*Transpose(T2aa(:,I,:,A)))&
                                            + Sum(ERIbb(C,vB:NBF,K,1:oB)*Transpose(T2ab(I,:,A,:))) 
          End Do
          End Do
        End Do
        End Do
!Construct Xbikac
        Xbikac = Zero
        Do I = 1, oB
        Do A = vB, NBF
          Do K = 1, oA
          Do C = vA, NBF
            Xbikac(I,K,A,C) = Xbikac(I,K,A,C) + Sum(ERIab(C,vB:NBF,K,1:oB)*Transpose(T2bb(:,I,:,A)))&
                                              + Sum(ERIaa(C,vA:NBF,K,1:oA)*Transpose(T2ab(:,I,:,A)))
          End Do
          End Do
        End Do
        End Do
!Use Xikac
        Do I = 1, oA
        Do A = vA, NBF
          G1a(I,A) = G1a(I,A) + Sum(Xikac(I,:,A,:)*(T1b(:,:)))
        End Do
        End Do
!Use Xbikac
        Do I = 1, oB
        Do A = vB, NBF
          G1b(I,A) = G1b(I,A) + Sum(Xbikac(I,:,A,:)*(T1a(:,:)))
        End Do
        End Do

!Construct Fik
        Fik = Zero
        Do I = 1, oA
        Do K = 1, oA
          Fik(I,K) = Fik(I,K) + Sum(FockA(K,vA:NBF)*T1a(I,:))  
        End Do
        End Do
        Fik = -Fik
!Construct Fbik
        Fbik = Zero
        Do I = 1, oB
        Do K = 1, oB
          Fbik(I,K) = Fbik(I,K) + Sum(FockB(K,vB:NBF)*T1b(I,:))  
        End Do
        End Do
        Fbik = -Fbik
!Use Fik
        Do I = 1, oA
        Do A = vA, NBF
          G1a(I,A) = G1a(I,A) + Sum(Fik(I,:)*T1a(:,A)) 
        End Do
        End Do
!Use Fbik
        Do I = 1, oB
        Do A = vB, NBF
          G1b(I,A) = G1b(I,A) + Sum(Fbik(I,:)*T1b(:,A))
        End Do
        End Do


!Construct Ica
        Ica = Zero
        Do C = vA, NBF
        Do A = vA, NBF
          Ica(C,A) = Ica(C,A) + Sum(ERIaa(C,vA:NBF,A,1:oA)*Transpose(T1a(:,:)))& 
                              + Sum(ERIab(C,vB:NBF,A,1:oB)*Transpose(T1b(:,:))) 
        End Do
        End Do
!Construct Ibca
        Ibca = Zero
        Do C = vB, NBF
        Do A = vB, NBF
          Ibca(C,A) = Ibca(C,A) + Sum(ERIbb(C,vB:NBF,A,1:oB)*Transpose(T1b(:,:)))&
                                + Sum(ERIab(vA:NBF,C,1:oA,A)*Transpose(T1a(:,:)))
        End Do
        End Do
!Use Ica
        Do I = 1, oA
        Do A = vA, NBF
          G1a(I,A) = G1a(I,A) + Sum(Ica(:,A)*T1a(I,:))
        End Do
        End Do
!Use Ibca
        Do I = 1, oB
        Do A = vB, NBF
          G1b(I,A) = G1b(I,A) + Sum(Ibca(:,A)*T1b(I,:))
        End Do
        End Do


!Construct Iik
        Iik = Zero
        Do I = 1, oA
        Do K = 1, oA
          Iik(I,K) = Iik(I,K) + Sum(ERIaa(I,vA:NBF,K,1:oA)*Transpose(T1a(:,:)))&
                              + Sum(ERIab(I,vB:NBF,K,1:oB)*Transpose(T1b(:,:)))
        End Do
        End Do
        Iik = -Iik
!Construct Ibik
        Ibik = Zero
        Do I = 1, oB
        Do K = 1, oB
          Ibik(I,K) = Ibik(I,K) + Sum(ERIbb(I,vB:NBF,K,1:oB)*Transpose(T1b(:,:)))&
                                + Sum(ERIab(vA:NBF,I,1:oA,K)*Transpose(T1a(:,:)))
        End Do
        End Do
        Ibik = -Ibik
!Use Iik
        Do I = 1, oA
        Do A = vA, NBF
          G1a(I,A) = G1a(I,A) + Sum(Iik(I,:)*T1a(:,A))
        End Do
        End Do
!Use Ibik
        Do I = 1, oB
        Do A = vB, NBF
          G1b(I,A) = G1b(I,A) + Sum(Ibik(I,:)*T1b(:,A)) 
        End Do
        End Do

!Construct Ick
        Ick = Zero
        Do C = vA, NBF
        Do K = 1, oA
          Ick(C,K) = Ick(C,K) + Sum(ERIaa(C,vA:NBF,K,1:oA)*Transpose(T1a(:,:)))&
                              + Sum(ERIab(C,vB:NBF,K,1:oB)*Transpose(T1b(:,:)))

        End Do
        End Do
        Ick = -Ick
!Construct Ibck
        Ibck = Zero
        Do C = vB, NBF
        Do K = 1, oB
          Ibck(C,K) = Ibck(C,K) + Sum(ERIbb(C,vB:NBF,K,1:oB)*Transpose(T1b(:,:)))&
                                + Sum(ERIab(vA:NBF,C,1:oA,K)*Transpose(T1a(:,:)))
        End Do
        End Do
        Ibck = -Ibck
!Use Ick
        Do I = 1, oA
        Do A = vA, NBF
         Do C = vA, NBF
          G1a(I,A) = G1a(I,A) + Sum(Ick(C,:)*T1a(:,A))*T1a(I,C)
         End Do
        End Do
        End Do
!Use Ibck
        Do I = 1, oB
        Do A = vB, NBF
          Do C = vB, NBF
          G1b(I,A) = G1b(I,A) + Sum(Ibck(C,:)*T1b(:,A))*T1b(I,C)
          End Do
        End Do
        End Do




       !debug closed shell
       !G1b = G1a

      deallocate(Jca,&
               Jbca,&
               Jik,&
               Jbik,&
               Wikac,&
               Wbikac,&
               Xikac,&
               Xbikac,&
               Fik,&
               Fbik,&
               Ica,& 
               Ibca,& 
               Iik,& 
               Ibik,& 
               Ick,& 
               Ibck) 

      End Subroutine GetG1



