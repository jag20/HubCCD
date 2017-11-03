!This fortran module contains the utilities for building the
!right-hand-side of the spin-summed unrestricted CCSD equations that can
!be used with arbitrary Hamiltonians (i.e. specific structure of
!integral is not assumed.

      Subroutine GetG2(G2aa,G2ab,G2bb,T2aa,T2ab,T2bb,ERIaa,ERIab,ERIbb,oa,ob,na,nb,NBF)
      Integer,          Intent(In)  :: oA, oB, nA, nB, NBF
      double precision, Intent(In)  :: ERIaa(NBF,NBF,NBF,NBF)
      double precision, Intent(In)  ::   ERIab(NBF,NBF,NBF,NBF)
      double precision, Intent(In)  ::   ERIbb(NBF,NBF,NBF,NBF)
      double precision, Intent(In)  :: T2aa(1:OA,1:OA,NA:NBF,NA:NBF)
      double precision, Intent(In)  :: T2ab(1:OA,1:OB,NA:NBF,NB:NBF)
      double precision, Intent(In)  :: T2bb(1:OB,1:OB,NB:NBF,NB:NBF)
      double precision, Intent(Out) :: G2aa(1:OA,1:OA,NA:NBF,NA:NBF)
      double precision, Intent(Out) :: G2ab(1:OA,1:OB,NA:NBF,NB:NBF)
      double precision, Intent(Out) :: G2bb(1:OB,1:OB,NB:NBF,NB:NBF)
! Indices
      Integer :: I, J, K, L, A, B, C, D
! Intermediates
      double precision :: Jijkl, Jaaaa, aaJaaaa, Jbbbb, bbJbbbb,&
                        Jovov, Jovvo, bbJovvo, Jvovo, aaJvoov, Jvoov,&
                        Zero =0.0e0, F12 = 0.50e0, F14 = 0.25e0
      double precision, Allocatable :: JooAA(:,:), JvvAA(:,:)
      double precision, Allocatable :: JooBB(:,:), JvvBB(:,:)
      double precision, Allocatable :: AAJooAA(:,:), AAJvvAA(:,:)
      double precision, Allocatable :: BBJooBB(:,:), BBJvvBB(:,:)
      !f2py depend(NBF) EriAA, EriAB, ERIbb
      !f2py depend(NBF,oA,na) T2aa, G2aa
      !f2py depend(NBF,ob,nb) T2bb, G2bb
      !f2py depend(NBF,oa,ob,na,nb) T2ab, G2ab

!================================================================!
!  Build the right-hand-sides G2aa, G2ab, G2bb.                  !
!                                                                !
!  To minimize storage of N^4 objects, I compute the various     !
!  intermediates in place, which does not help readability!      !
!----------------------------------------------------------------!
!  Start by getting the driver, which only contributes to G2ab.  !
!================================================================!
! This code is based on a Hubbard-Specific CCD code by Tom       !
! Henderson. Modified by jag20 from th4 for use with arbitrary   !
! Hamiltonians 7.29.15                                           !
!================================================================!

! Set up some things 
      Allocate(JooAA(1:oA,1:oA),             &
               AAJooAA(1:oA,1:oA),           &
               JooBB(1:oB,1:oB),             &
               BBJooBB(1:oB,1:oB),           &
               JvvAA(nA:NBF,nA:NBF),             &
               AAJvvAA(nA:NBF,nA:NBF),           &
               JvvBB(nB:NBF,nB:NBF),             &
               BBJvvBB(nB:NBF,nB:NBF))

!=============================!
! Get the drivers             !
!=============================!

      G2aa = ERIaa(1:oA,1:oA,nA:NBF,nA:NBF)
      G2ab = ERIab(1:oA,1:oB,nA:NBF,nB:NBF)
      G2bb = ERIbb(1:oB,1:oB,nB:NBF,nB:NBF)



!=============================!
! Get the ladders             !
!=============================!

!aa
      Do I = 1,oA
      Do J = 1,oA
        Do A = nA,NBF
        Do B = nA,NBF
          G2aa(I,J,A,B) = G2aa(I,J,A,B) + Sum(ERIaa(nA:NBF,nA:NBF,A,B)*T2aa(I,J,:,:))*F12  &
                                        + Sum(ERIaa(I,J,1:oA,1:oA)*T2aa(:,:,A,B))*F12
        End Do
        End Do
! Quadratic pieces
        Do K = 1,oA
        Do L = 1,oA
          Jijkl = Sum(ERIaa(nA:NBF,nA:NBF,K,L)*T2aa(I,J,:,:))
          Do A = nA,NBF
          Do B = nA,NBF
            G2aa(I,J,A,B) = G2aa(I,J,A,B) + Jijkl*T2aa(K,L,A,B)*F14
          End Do
          End Do
        End Do
        End Do
      End Do
      End Do

!ab
      Do I = 1,oA
      Do J = 1,oB
        Do A = nA,NBF
        Do B = nB,NBF
          G2ab(I,J,A,B) = G2ab(I,J,A,B) + Sum(ERIab(nA:NBF,nB:NBF,A,B)*T2ab(I,J,:,:))  &
                                        + Sum(ERIab(I,J,1:oA,1:oB)*T2ab(:,:,A,B))
        End Do
        End Do
! Quadratic pieces
        Do K = 1,oA
        Do L = 1,oB
          Jijkl = Sum(ERIab(nA:NBF,nB:NBF,K,L)*T2ab(I,J,:,:))
          Do A = nA,NBF
          Do B = nB,NBF
            G2ab(I,J,A,B) = G2ab(I,J,A,B) + Jijkl*T2ab(K,L,A,B)
          End Do
          End Do
        End Do
        End Do
      End Do
      End Do
!bb
      Do I = 1,oB
      Do J = 1,oB
        Do A = nB,NBF
        Do B = nB,NBF
          G2bb(I,J,A,B) = G2bb(I,J,A,B) + Sum(ERIbb(nB:NBF,nB:NBF,A,B)*T2bb(I,J,:,:))*F12  &
                                        + Sum(ERIbb(I,J,1:oB,1:oB)*T2bb(:,:,A,B))*F12
        End Do
        End Do
! Quadratic pieces
        Do K = 1,oB
        Do L = 1,oB
          Jijkl = Sum(ERIbb(nB:NBF,nB:NBF,K,L)*T2bb(I,J,:,:))
          Do A = nB,NBF
          Do B = nB,NBF
            G2bb(I,J,A,B) = G2bb(I,J,A,B) + Jijkl*T2bb(K,L,A,B)*F14
          End Do
          End Do
        End Do
        End Do
      End Do
      End Do




!=============================!
!  Now get the mosaic terms.  !
!=============================!

! Compute the intermediates JooAA, JooBB, JvvAA, JvvBB
! as well as           AAJooAA, BBJooBB, AAJvvAA, BBJvvBB
      JooAA = Zero
      JvvAA = Zero
      Do L = 1,oB
      Do D = nB,NBF
        Do I = 1,oA
        Do J = 1,oA
        Do C = nA,NBF
          JooAA(I,J) = JooAA(I,J) + ERIab(C,D,J,L)*T2ab(I,L,C,D)
        End Do
        End Do
        End Do

        Do A = nA,NBF
        Do B = nA,NBF
        Do K = 1,oA
          JvvAA(A,B) = JvvAA(A,B) - ERIab(A,D,K,L)*T2ab(K,L,B,D)
        End Do
        End Do
        End Do
      End Do
      End Do

      AAJooAA = Zero
      AAJvvAA = Zero
      Do L = 1,oA
      Do D = nA,NBF
        Do I = 1,oA
        Do J = 1,oA
        Do C = nA,NBF
          AAJooAA(I,J) = AAJooAA(I,J) + ERIaa(C,D,J,L)*T2aa(I,L,C,D)*F12
        End Do
        End Do
        End Do

        Do A = nA,NBF
        Do B = nA,NBF
        Do K = 1,oA
          AAJvvAA(A,B) = AAJvvAA(A,B) - ERIaa(A,D,K,L)*T2aa(K,L,B,D)*F12
        End Do
        End Do
        End Do
      End Do
      End Do


      JooBB = Zero
      JvvBB = Zero
      Do L = 1,oA
      Do D = nA,NBF
        Do I = 1,oB
        Do J = 1,oB
        Do C = nB,NBF
          JooBB(I,J) = JooBB(I,J) + ERIab(D,C,L,J)*T2ab(L,I,D,C)
        End Do
        End Do
        End Do

        Do A = nB,NBF
        Do B = nB,NBF
        Do K = 1,oB
          JvvBB(A,B) = JvvBB(A,B) - ERIab(D,A,L,K)*T2ab(L,K,D,B)
        End Do
        End Do
        End Do
      End Do
      End Do

      BBJooBB = Zero
      BBJvvBB = Zero
      Do L = 1,oB
      Do D = nB,NBF
        Do I = 1,oB
        Do J = 1,oB
        Do C = nB,NBF
          BBJooBB(I,J) = BBJooBB(I,J) + ERIbb(C,D,J,L)*T2bb(I,L,C,D)*F12
        End Do
        End Do
        End Do

        Do A = nB,NBF
        Do B = nB,NBF
        Do K = 1,oB
          BBJvvBB(A,B) = BBJvvBB(A,B) - ERIbb(A,D,K,L)*T2bb(K,L,B,D)*F12
        End Do
        End Do
        End Do
      End Do
      End Do


! Use the intermediates
      Do I = 1,oA
      Do J = 1,oB
      Do A = nA,NBF
      Do B = nB,NBF
        G2ab(I,J,A,B) = G2ab(I,J,A,B)                                                                   &
                      + Dot_Product((AAJvvAA(:,A)+JvvAA(:,A)),T2ab(I,J,:,B))                            &
                      + Dot_Product((BBJvvBB(:,B)+JvvBB(:,B)),T2ab(I,J,A,:))                            &
                      - Dot_Product((AAJooAA(I,:)+JooAA(I,:)),T2ab(:,J,A,B))                            &
                      - Dot_Product((BBJooBB(J,:)+JooBB(J,:)),T2ab(I,:,A,B))
      End Do
      End Do
      End Do
      End Do

      Do I = 1,oA
      Do J = 1,oA
      Do A = nA,NBF
      Do B = nA,NBF
        G2aa(I,J,A,B) = G2aa(I,J,A,B)                                                                   &
                      + Dot_Product((AAJvvAA(:,A)+JvvAA(:,A)),T2aa(I,J,:,B))                            &
                      + Dot_Product((AAJvvAA(:,B)+JvvAA(:,B)),T2aa(I,J,A,:))                            &
                      - Dot_Product((AAJooAA(I,:)+JooAA(I,:)),T2aa(:,J,A,B))                            &
                      - Dot_Product((AAJooAA(J,:)+JooAA(J,:)),T2aa(I,:,A,B))
      End Do
      End Do
      End Do
      End Do
   
      Do I = 1,oB
      Do J = 1,oB
      Do A = nB,NBF
      Do B = nB,NBF
        G2bb(I,J,A,B) = G2bb(I,J,A,B)                                                                   &
                      + Dot_Product((BBJvvBB(:,A)+JvvBB(:,A)),T2bb(I,J,:,B))                            &
                      + Dot_Product((BBJvvBB(:,B)+JvvBB(:,B)),T2bb(I,J,A,:))                            &
                      - Dot_Product((BBJooBB(I,:)+JooBB(I,:)),T2bb(:,J,A,B))                            &
                      - Dot_Product((BBJooBB(J,:)+JooBB(J,:)),T2bb(I,:,A,B))
      End Do
      End Do
      End Do
      End Do




!==============================================================!
!  Now the ring terms.                                         !
!==============================================================!

! Terms which use Jaaaa and aaJaaaa - I'll calculate Jaaaa(ICAK)
      Do I = 1,oA
      Do K = 1,oA
      Do A = nA,NBF
      Do C = nA,NBF
        Jaaaa = Zero
        Do L = 1,oB
        Do D = nB,NBF
          Jaaaa = Jaaaa + ERIab(C,D,K,L)*T2ab(I,L,A,D)
        End Do
        End Do
        Jaaaa = Jaaaa*F12
        aaJaaaa = Zero
        Do L = 1,oA
        Do D = nA,NBF
          aaJaaaa = aaJaaaa + ERIaa(C,D,K,L)*T2aa(I,L,A,D)
        End Do
        End Do
        aaJaaaa = aaJaaaa*F12

! Shows up in G2ab
        Do J = 1,oB
        Do B = nB,NBF
          G2ab(I,J,A,B) = G2ab(I,J,A,B) + (ERIaa(I,C,A,K)+aaJaaaa+Jaaaa)*T2ab(K,J,C,B)
        End Do
        End Do

! Also shows up in G2aa.  Update several RHSs for one intermediate
        Do J = 1,oA
        Do B = nA,NBF
          G2aa(I,J,A,B) = G2aa(I,J,A,B) + (ERIaa(I,C,A,K)+aaJaaaa+Jaaaa)*T2aa(J,K,B,C)
          G2aa(J,I,B,A) = G2aa(J,I,B,A) + (ERIaa(I,C,A,K)+aaJaaaa+Jaaaa)*T2aa(J,K,B,C)
          G2aa(I,J,B,A) = G2aa(I,J,B,A) - (ERIaa(I,C,A,K)+aaJaaaa+Jaaaa)*T2aa(J,K,B,C)
          G2aa(J,I,A,B) = G2aa(J,I,A,B) - (ERIaa(I,C,A,K)+aaJaaaa+Jaaaa)*T2aa(J,K,B,C)
        End Do
        End Do
      End Do
      End Do
      End Do
      End Do




! Terms which use Jbbbb - I'll calculate Jbbbb(JCBK)
      Do J = 1,oB
      Do K = 1,oB
      Do B = nB,NBF
      Do C = nB,NBF
        Jbbbb = Zero
        Do L = 1,oA
        Do D = nA,NBF
          Jbbbb = Jbbbb + ERIab(D,C,L,K)*T2ab(L,J,D,B)
        End Do
        End Do
        Jbbbb = Jbbbb*F12
        bbJbbbb = Zero
        Do L = 1,oB
        Do D = nB,NBF
          bbJbbbb = bbJbbbb + ERIbb(C,D,K,L)*T2bb(J,L,B,D)
        End Do
        End Do
        bbJbbbb = bbJbbbb*F12

! Shows up in G2ab
        Do I = 1,oA
        Do A = nA,NBF
          G2ab(I,J,A,B) = G2ab(I,J,A,B) + (ERIbb(J,C,B,K)+bbJbbbb+Jbbbb)*T2ab(I,K,A,C)
        End Do
        End Do

! Also shows up in G2bb.  Update several RHSs for one intermediate
        Do I = 1,oB
        Do A = nB,NBF
          G2bb(I,J,A,B) = G2bb(I,J,A,B) + (ERIbb(J,C,B,K)+bbJbbbb+Jbbbb)*T2bb(I,K,A,C)
          G2bb(J,I,B,A) = G2bb(J,I,B,A) + (ERIbb(J,C,B,K)+bbJbbbb+Jbbbb)*T2bb(I,K,A,C)
          G2bb(I,J,B,A) = G2bb(I,J,B,A) - (ERIbb(J,C,B,K)+bbJbbbb+Jbbbb)*T2bb(I,K,A,C)
          G2bb(J,I,A,B) = G2bb(J,I,A,B) - (ERIbb(J,C,B,K)+bbJbbbb+Jbbbb)*T2bb(I,K,A,C)
        End Do
        End Do
      End Do
      End Do
      End Do
      End Do




! Terms which use Jovvo - I'll calculate Jovvo(ICAK)
      Do I = 1,oA
      Do K = 1,oB
      Do A = nA,NBF
      Do C = nB,NBF
        Jovvo = 2*ERIab(I,C,A,K)
        Do L = 1,oA
        Do D = nA,NBF
          Jovvo = Jovvo + ERIab(D,C,L,K)*T2aa(I,L,A,D)
        End Do
        End Do
        Jovvo = Jovvo*F12
        bbJovvo = Zero
        Do L = 1,oB
        Do D = nB,NBF
          bbJovvo = bbJovvo + ERIbb(C,D,K,L)*T2ab(I,L,A,D)
        End Do
        End Do
        bbJovvo = bbJovvo*F12

! Shows up in G2ab
        Do J = 1,oB
        Do B = nB,NBF
          G2ab(I,J,A,B) = G2ab(I,J,A,B) + (bbJovvo+Jovvo)*T2bb(K,J,C,B)
        End Do
        End Do
         
! Also shows up in G2aa.  Update several RHSs for one intermediate
        Do J = 1,oA
        Do B = nA,NBF
          G2aa(I,J,A,B) = G2aa(I,J,A,B) + (bbJovvo+Jovvo)*T2ab(J,K,B,C)
          G2aa(J,I,B,A) = G2aa(J,I,B,A) + (bbJovvo+Jovvo)*T2ab(J,K,B,C)
          G2aa(I,J,B,A) = G2aa(I,J,B,A) - (bbJovvo+Jovvo)*T2ab(J,K,B,C)
          G2aa(J,I,A,B) = G2aa(J,I,A,B) - (bbJovvo+Jovvo)*T2ab(J,K,B,C)
        End Do
        End Do
      End Do
      End Do
      End Do
      End Do




! Terms which use Jvoov - I'll calculate Jvoov(CJKB)
      Do J = 1,oB
      Do K = 1,oA
      Do B = nB,NBF
      Do C = nA,NBF
        Jvoov = ERIab(C,J,K,B)*2
        Do L = 1,oB
        Do D = nB,NBF
          Jvoov = Jvoov + ERIab(C,D,K,L)*T2bb(J,L,B,D)
        End Do
        End Do
        Jvoov = Jvoov*F12

        aaJvoov = Zero
        Do L = 1,oA
        Do D = nA,NBF
          aaJvoov = aaJvoov + ERIaa(C,D,K,L)*T2ab(L,J,D,B)
        End Do
        End Do
        aaJvoov = aaJvoov*F12


! Shows up in G2ab
        Do I = 1,oA
        Do A = nA,NBF
          G2ab(I,J,A,B) = G2ab(I,J,A,B) + (aaJvoov+Jvoov)*T2aa(I,K,A,C)
        End Do
        End Do

! Also shows up in G2bb.  Update several RHSs for one intermediate
        Do I = 1,oB
        Do A = nB,NBF
          G2bb(I,J,A,B) = G2bb(I,J,A,B) + (aaJvoov+Jvoov)*T2ab(K,I,C,A)
          G2bb(J,I,B,A) = G2bb(J,I,B,A) + (aaJvoov+Jvoov)*T2ab(K,I,C,A)
          G2bb(I,J,B,A) = G2bb(I,J,B,A) - (aaJvoov+Jvoov)*T2ab(K,I,C,A)
          G2bb(J,I,A,B) = G2bb(J,I,A,B) - (aaJvoov+Jvoov)*T2ab(K,I,C,A)
        End Do
        End Do
      End Do
      End Do
      End Do
      End Do




! There's a G2ab term which uses Jovov(ICKB)
      Do I = 1,oA
      Do K = 1,oA
      Do B = nB,NBF
      Do C = nB,NBF
        Jovov = 2*ERIab(I,C,K,B)
        Do L = 1,oB
        Do D = nA,NBF
          Jovov = Jovov - ERIab(D,C,K,L)*T2ab(I,L,D,B)
        End Do
        End Do
        Jovov = Jovov*F12

        Do J = 1,oB
        Do A = nA,NBF
          G2ab(I,J,A,B) = G2ab(I,J,A,B) - Jovov*T2ab(K,J,A,C)
        End Do
        End Do
      End Do
      End Do
      End Do
      End Do




! There's a G2ab term which uses Jvovo(CJAK)
      Do J = 1,oB
      Do K = 1,oB
      Do A = nA,NBF
      Do C = nA,NBF
        Jvovo = 2*ERIab(C,J,A,K)
        Do L = 1,oA
        Do D = nB,NBF
          Jvovo = Jvovo - ERIab(C,D,L,K)*T2ab(L,J,A,D)
        End Do
        End Do
        Jvovo = Jvovo*F12

        Do I = 1,oA
        Do B = nB,NBF
          G2ab(I,J,A,B) = G2ab(I,J,A,B) - Jvovo*T2ab(I,K,C,B)
        End Do
        End Do
      End Do
      End Do
      End Do
      End Do
!      print *, "T2aa", G2aa
!      print *, "T2bb", G2bb
 

      deallocate(JooAA,&
               AAJooAA,&
               JooBB,&
               BBJooBB,&
               JvvAA,&
               AAJvvAA,&
               JvvBB,&
               BBJvvBB)


      End Subroutine GetG2

      Subroutine GetCCSDG2(G2aa,G2ab,G2bb,T2aa,T2ab,T2bb,T1a,T1b,FockA,FockB,ERIaa,ERIab,ERIbb,NOccA,NOccB,NBF)
      !=====================================================================!
      ! This subroutine gets CCSD T2 amplitudes.
      ! Suitable for molecules.
      ! -jag20, 8.06.15
      !=====================================================================!
      Integer,        Intent(In)  :: NOccA, NOccB, NBF
      double precision, Intent(In)  :: ERIaa(NBF,NBF,NBF,NBF),&
                                     ERIab(NBF,NBF,NBF,NBF),&
                                     ERIbb(NBF,NBF,NBF,NBF),&
                                     FockA(NBF,NBF),FockB(NBF,NBF)
      double precision, Intent(In)  :: &
                                     T1a(1:NOccA,NOccA+1:NBF),&
                                     T1b(1:NOccB,NOccB+1:NBF)
      double precision, Intent(In)  :: T2aa(1:NOccA,1:NOccA,NOccA+1:NBF,NOccA+1:NBF)
      double precision, Intent(In)  :: T2ab(1:NOccA,1:NOccB,NOccA+1:NBF,NOccB+1:NBF)
      double precision, Intent(In)  :: T2bb(1:NOccB,1:NOccB,NOccB+1:NBF,NOccB+1:NBF)
      double precision, Intent(Out) :: G2aa(1:NOccA,1:NOccA,NOccA+1:NBF,NOccA+1:NBF)
      double precision, Intent(Out) :: G2ab(1:NOccA,1:NOccB,NOccA+1:NBF,NOccB+1:NBF)
      double precision, Intent(Out) :: G2bb(1:NOccB,1:NOccB,NOccB+1:NBF,NOccB+1:NBF)

      !f2py depend(NBF) FockA, FockB, EriAA, EriAB, ERIbb, T2aa, T2ab,T2bb, T1a, T1b, G2aa, G2ab, G2bb
      !f2py depend(NoccA) T2aa, T2ab, T1a, G2aa, G2ab
      !f2py depend(NoccB) T2ab, T2bb, T1b, G2ab, G2bb
! Indices
      Integer :: I, J, K, L, A, B, C, D, vA,vB, oA, oB

!other local
      double precision :: Zero =0.0e0, F12 = 0.50e0, F14 = 0.25e0
      double precision, allocatable :: Fil(:,:),Fad(:,:), Fkj(:,:), Fcb(:,:),&
                                  wilad(:,:,:,:),wikdb(:,:,:,:),wljac(:,:,:,:),wkjdb(:,:,:,:),&
                                  wcdab(:,:,:,:),wijkl(:,:,:,:),&
                                  mikad(:,:,:,:),mikbd(:,:,:,:),mjkad(:,:,:,:),mjlbc(:,:,:,:),&
                                  Fbil(:,:),Fbad(:,:), Fbkj(:,:), Fbcb(:,:),&
                                  wbilad(:,:,:,:),wbikdb(:,:,:,:),wbljac(:,:,:,:),wbkjdb(:,:,:,:),&
                                  wbcdab(:,:,:,:),wbijkl(:,:,:,:),&
                                  mbikad(:,:,:,:),mbikbd(:,:,:,:),mbjkad(:,:,:,:),mbjlbc(:,:,:,:),&
                                  !alpha/beta
                                  Kjkad(:,:,:,:),Kcdab(:,:,:,:),Kklij(:,:,:,:),Kildb(:,:,:,:),Rljdb(:,:,:,:)
     
      vA = noccA+1
      vB = noccB+1
      oA = noccA
      oB = noccB
                                  

!Allocate stuff
         Allocate(Fil(1:oA,1:oA),               &
               Fad(vA:NBF,vA:NBF),               &
               Fkj(1:oA,1:oA),               &
               Fcb(vA:NBF,vA:NBF),               &
               wilad(1:oa,1:oa,va:NBF,va:NBF),   &
               wikdb(1:oa,1:oa,va:NBF,va:NBF),   &
               wljac(1:oa,1:oa,va:NBF,va:NBF),   &
               wkjdb(1:oa,1:oa,va:NBF,va:NBF),   &
               wcdab(va:NBF,va:NBF,va:NBF,va:NBF),   &
               wijkl(1:oa,1:oa,1:oa,1:oa),   &
               mikad(1:oa,1:ob,va:NBF,vb:NBF),   &
               mikbd(1:oa,1:ob,va:NBF,vb:NBF),   &
               mjkad(1:oa,1:ob,va:NBF,vb:NBF),   &
               mjlbc(1:oa,1:ob,va:NBF,vb:NBF),   &
               Fbil(1:ob,1:ob),               &
               Fbad(vb:NBF,vb:NBF),               &
               Fbkj(1:ob,1:ob),               &
               Fbcb(vb:NBF,vb:NBF),               &
               wbijkl(1:ob,1:ob,1:ob,1:ob),   &
               wbilad(1:ob,1:ob,vb:NBF,vb:NBF),   &
               wbikdb(1:ob,1:ob,vb:NBF,vb:NBF),   &
               wbljac(1:ob,1:ob,vb:NBF,vb:NBF),   &
               wbkjdb(1:ob,1:ob,vb:NBF,vb:NBF),   &
               wbcdab(vb:NBF,vb:NBF,vb:NBF,vb:NBF),   &
               mbikad(1:ob,1:oa,vb:NBF,va:NBF),   &
               mbikbd(1:ob,1:oa,vb:NBF,va:NBF),   &
               mbjkad(1:ob,1:oa,vb:NBF,va:NBF),   &
               mbjlbc(1:ob,1:oa,vb:NBF,va:NBF),   &
               Kjkad(1:ob,1:ob,va:NBF,va:NBF),    &
               Kildb(1:oa,1:oa,vb:NBF,vb:NBF),    &
               Rljdb(1:oa,1:ob,va:NBF,vb:NBF),    &
               Kcdab(va:NBF,vb:NBF,va:NBF,vb:NBF),    &
               Kklij(1:oa,1:ob,1:oa,1:ob))


       ! print *, "Inside GetCCSDG2"
! First get amplitude contributions that do not involve singles.
!       Call GetG2(G2aa,G2ab,G2bb,T2aa,T2ab,T2bb,ERIaa,ERIab,ERIbb,NOccA,NOccB,NBF)
       Call GetG2(G2aa,G2ab,G2bb,T2aa,T2ab,T2bb,ERIaa,ERIab,ERIbb,oa,ob,va,vb,NBF)

! We will follow my notes and worked out intermediates. Apologies if you do not
! have the Notes. We will first get all single amplitude involving terms not
! attached to an intermediate.
! For alpha/beta amplitudes, we will re-use and modify pre-existing intermediates where
! possible

!!First we get alpha/alpha contributions
        Do I = 1, NOccA
        Do J = 1, NOccA
        Do A = NOccA+1, NBF
        Do B = NOccA+1, NBF
          G2aa(I,J,A,B) = G2aa(I,J,A,B) + Sum(ERIaa(NOccA+1:NBF,J,A,B)*T1a(I,:)) - Sum(ERIaa(NOccA+1:NBF,I,A,B)*T1a(J,:))&
                                        - Sum(ERIaa(I,J,1:NOccA,B)*T1a(:,A)) + Sum(ERIaa(I,J,1:NOccA,A)*T1a(:,B))
          
          Do K = 1, NOccA
            G2aa(I,J,A,B) = G2aa(I,J,A,B) + Sum(ERIaa(I,J,K,1:NOccA)*T1a(:,B))*T1a(K,A)
          End Do
          Do D = NOccA+1, NBF
            G2aa(I,J,A,B) = G2aa(I,J,A,B) + Sum(ERIaa(NOccA+1:NBF,D,A,B)*T1a(I,:))*T1a(J,D)
          End Do
!          
          Do K = 1, NOccA
            G2aa(I,J,A,B) = G2aa(I,J,A,B) - Sum(ERIaa(NOccA+1:NBF,J,K,B)*T1a(I,:))*T1a(K,A)&
                                          + Sum(ERIaa(NOccA+1:NBF,J,K,A)*T1a(I,:))*T1a(K,B)&
                                          + Sum(ERIaa(NOccA+1:NBF,I,K,B)*T1a(J,:))*T1a(K,A)&
                                          - Sum(ERIaa(NOccA+1:NBF,I,K,A)*T1a(J,:))*T1a(K,B)
          End Do

          Do K = 1, NOccA
          Do D = NOccA+1, NBF
            G2aa(I,J,A,B) = G2aa(I,J,A,B) - Sum(ERIaa(NOccA+1:NBF,D,K,B)*T1a(I,:))*T1a(K,A)*T1a(J,D)&
                                          + Sum(ERIaa(NOccA+1:NBF,D,K,A)*T1a(I,:))*T1a(K,B)*T1a(J,D) 
          End Do
          End Do

            Do K = 1, NOccA
            Do L = 1, NOccA
              G2aa(I,J,A,B) = G2aa(I,J,A,B) + Sum(ERIaa(NOccA+1:NBF,J,K,L)*T1a(I,:))*T1a(K,A)*T1a(L,B)& 
                                            - Sum(ERIaa(NOccA+1:NBF,I,K,L)*T1a(J,:))*T1a(K,A)*T1a(L,B)
            End Do
            End Do

          Do K = 1, NOccA
          Do L = 1, NOccA
          Do D = NOccA+1, NBF
            G2aa(I,J,A,B) = G2aa(I,J,A,B) + Sum(ERIaa(NOccA+1:NBF,D,K,L)*T1a(I,:))*T1a(J,D)*T1a(K,A)*T1a(L,B)
          End Do
          End Do
          End Do


        End Do
        End Do
        End Do
        End Do
!!
!!!Here come the beta/beta terms not involving intermediates
        Do I = 1, NOccb
        Do J = 1, NOccb
        Do A = NOccb+1, NBF
        Do B = NOccb+1, NBF
          G2bb(I,J,A,B) = G2bb(I,J,A,B) + Sum(ERIbb(noccb+1:nbf,J,A,B)*T1b(I,:)) - Sum(ERIbb(noccb+1:nbf,I,A,B)*T1b(J,:))&
                                        - Sum(ERIbb(I,J,1:noccb,B)*T1b(:,A)) + Sum(ERIbb(I,J,1:noccb,A)*T1b(:,B))
         
          Do K = 1, NOccb
            G2bb(I,J,A,B) = G2bb(I,J,A,B) + Sum(ERIbb(I,J,K,1:noccb)*T1b(:,B))*T1b(K,A)
          End Do
          Do D = NOccb+1, NBF
           G2bb(I,J,A,B) = G2bb(I,J,A,B) + Sum(ERIbb(NOccb+1:NBF,D,A,B)*T1b(I,:))*T1b(J,D)
          End Do
!          
          Do K = 1, NOccb
            G2bb(I,J,A,B) = G2bb(I,J,A,B) - Sum(ERIbb(NOccb+1:NBF,J,K,B)*T1b(I,:))*T1b(K,A)&
                                          + Sum(ERIbb(NOccb+1:NBF,J,K,A)*T1b(I,:))*T1b(K,B)&
                                          + Sum(ERIbb(NOccb+1:NBF,I,K,B)*T1b(J,:))*T1b(K,A)&
                                          - Sum(ERIbb(NOccb+1:NBF,I,K,A)*T1b(J,:))*T1b(K,B)
          End Do
!
          Do K = 1, NOccb
          Do D = NOccb+1, NBF
            G2bb(I,J,A,B) = G2bb(I,J,A,B) - Sum(ERIbb(NOccb+1:NBF,D,K,B)*T1b(I,:))*T1b(K,A)*T1b(J,D)&
                                          + Sum(ERIbb(NOccb+1:NBF,D,K,A)*T1b(I,:))*T1b(K,B)*T1b(J,D) 
          End Do
          End Do

          Do K = 1, NOccb
          Do L = 1, NOccb
            G2bb(I,J,A,B) = G2bb(I,J,A,B) + Sum(ERIbb(NOccb+1:NBF,J,K,L)*T1b(I,:))*T1b(K,A)*T1b(L,B)& 
                                          - Sum(ERIbb(NOccb+1:NBF,I,K,L)*T1b(J,:))*T1b(K,A)*T1b(L,B)
          End Do
          End Do
!
          Do L = 1, NOccb
          Do K = 1, NOccb
          Do D = NOccb+1, NBF
            G2bb(I,J,A,B) = G2bb(I,J,A,B) + Sum(ERIbb(NOccb+1:NBF,D,K,L)*T1b(I,:))*T1b(J,D)*T1b(K,A)*T1b(L,B)
          End Do
          End Do
          End Do

!!!
        End Do
        End Do
        End Do
        End Do
!
!!
!!!Here come alpha/beta not involving intermediates
        Do I = 1, nocca
        Do j = 1, noccb
        do a = nocca+1, nbf
        do b = noccb+1, nbf
          G2ab(i,j,a,b) = G2ab(i,j,a,b) + sum(ERIab(nocca+1:nbf,J,A,B)*T1a(I,:))&
                                        + sum(ERIab(I,noccb+1:nbf,A,B)*T1b(J,:))&
                                        - sum(ERIab(I,J,1:nocca,B)*T1a(:,A))&
                                        - sum(ERIab(I,J,A,1:noccb)*T1b(:,B))
          do k = 1, noccb
            G2ab(i,j,a,b) = G2ab(i,j,a,b) + sum(ERIab(I,J,1:nocca,k)*t1a(:,A))*t1b(k,B)&
                                          - sum(ERIab(nocca+1:nbf,J,A,K)*t1a(I,:))*t1b(K,B)&
                                          - sum(ERIab(I,noccb+1:nbf,A,K)*t1b(J,:))*t1b(K,B)
          end do
          do k = 1, nocca
            G2ab(i,j,a,b) = G2ab(i,j,a,b) - sum(ERIab(nocca+1:nbf,J,K,B)*t1a(I,:))*t1a(K,A)&
                                          - sum(ERIab(I,noccb+1:nbf,K,B)*t1b(J,:))*t1a(K,A) 
          end do
!
          do k = 1, nocca
          do l = 1, noccb
            G2ab(i,j,a,b) = G2ab(i,j,a,b) + sum(ERIab(I,noccb+1:nbf,K,l)*t1b(J,:))*t1a(K,A)*t1b(l,B)&
                                          + sum(ERIab(nocca+1:nbf,J,K,l)*t1a(I,:))*t1a(K,A)*t1b(l,B)
          end do 
          end do 
!!!problem with using here....
          do d = noccb+1, nbf
            G2ab(i,j,a,b) = G2ab(i,j,a,b) + sum(ERIab(nocca+1:nbf,D,A,B)*t1a(I,:))*t1b(J,D)
          end do

          do k = 1, nocca
          do d = noccb+1, nbf
            !G2ab(i,j,a,b) = G2ab(i,j,a,b) + sum(ERIab(nocca+1:nbf,D,K,B)*t1a(I,:))*t1a(K,A)*t1b(J,D)
            !potential sign problem
            G2ab(i,j,a,b) = G2ab(i,j,a,b) - sum(ERIab(nocca+1:nbf,D,K,B)*t1a(I,:))*t1a(K,A)*t1b(J,D)
          end do
          end do 

          do k = 1, noccb
          do d = noccb+1, nbf
            !G2ab(i,j,a,b) = G2ab(i,j,a,b) + sum(ERIab(nocca+1:nbf,D,A,K)*t1a(I,:))*t1b(K,B)*t1b(J,D)
            !potential sign problem
            G2ab(i,j,a,b) = G2ab(i,j,a,b) - sum(ERIab(nocca+1:nbf,D,A,K)*t1a(I,:))*t1b(K,B)*t1b(J,D)
          end do
          end do
!!.... to here

          do d = noccb+1, nbf
          do k = 1, nocca
          do l = 1, noccb 
            G2ab(i,j,a,b) = G2ab(i,j,a,b) + sum(ERIab(nocca+1:nbf,D,K,L)*t1a(I,:))*t1b(J,D)*t1a(K,A)*t1b(L,B)
          end do
          end do
          end do


        End Do
        End Do
        End Do
        End Do
!!!
!!!        Print *, "CCSD aa Double after no intermedate      : ", G2aa
!!!        Print *, "CCSD bb Double after no intermedate      : ", G2bb
!!!        Print *, "CCSD ab Double after no intermedate      : ", G2ab
!!!!=========================================================================================!
!!!! We will now get the alpha/alpha energies involving the identified
!!!! intermediates
!!!!=========================================================================================!
!!
!!!Make and use F intermediates
!!!alpha/alpha
         Fil = Zero 
         Do I = 1, NOccA
         Do L = 1, NOccA
          Fil(I,L) = Fil(I,L) + Sum(FockA(L,NOccA+1:NBF)*T1a(I,:))
           Do D = NOccA+1, NBF
           Do K = 1, NOccA
             Fil(I,L) = Fil(I,L) + Sum(ERIaa(NOccA+1:NBF,D,K,L)*T1a(K,:))*T1a(I,D)
           End Do
           Do K = 1, NOccB
             Fil(I,L) = Fil(I,L) + Sum(ERIab(D,NOccB+1:NBF,L,K)*T1b(K,:))*T1a(I,D)
 
           End Do
           End Do
           Do K = 1, NOccA
             Fil(I,L) = Fil(I,L) + Sum(ERIaa(NOccA+1:NBF,I,K,L)*T1a(K,:))
           End Do
           Do K = 1, NOccB
             Fil(I,L) = Fil(I,L) + Sum(ERIab(I,NOccB+1:NBF,L,K)*T1b(K,:))
           End Do
        End Do
        End Do
        Fil = -Fil
!!
!!
        Do I = 1, NOccA 
        Do J = 1, NOccA 
        Do A = NOccA+1, NBF 
        Do B = NOccA+1, NBF 
          G2aa(I,J,A,B) = G2aa(I,J,A,B) + Sum(Fil(I,:)*T2aa(:,J,A,B))
        End Do
        End Do
        End Do
        End Do
!!
        Fad = Zero
        Do A = NOccA+1, NBF
        Do D = NOccA+1, NBF
          Fad(A,D) = Fad(A,D) + Sum(FockA(1:NOccA,D)*T1a(:,A))
           Do L = 1, NOccA
            Do K = 1, NOccA
             Fad(A,D) = Fad(A,D) + Sum(ERIaa(NOccA+1:NBF,D,K,L)*T1a(K,:))*T1a(L,A)
           End Do    
           Do K = 1, NOccB
             Fad(A,D) = Fad(A,D) + Sum(ERIab(D,NOccB+1:NBF,L,K)*T1b(K,:))*T1a(L,A)
           End Do
           End Do
           Do K = 1, NOccA
            Fad(A,D) = Fad(A,D) - Sum(ERIaa(NOccA+1:NBF,D,K,A)*T1a(K,:))
           End Do
           Do K = 1, NOccB
            Fad(a,d) = fad(a,d) - sum(ERIab(D,NOccB+1:NBF,A,K)*T1b(K,:))
           End Do
        End Do
        End Do
        Fad = -Fad
! 
        Do I = 1, NOccA 
        Do J = 1, NOccA 
        Do A = NOccA+1, NBF 
        Do B = NOccA+1, NBF 
          G2aa(I,J,A,B) = G2aa(I,J,A,B) + Sum(Fad(A,:)*T2aa(I,J,:,B))
        End Do
        End Do
        End Do
        End Do
! 
! 
        Fkj = Zero
        Do K = 1, NOccA
        Do J = 1, NOccA
          Fkj(K,J) = Fkj(K,J) + Sum(FockA(K,NOccA+1:NBF)*T1a(J,:))
           Do D = NOccA+1, NBF
           Do C = NOccA+1, NBF
             Fkj(K,J) = Fkj(K,J) + Sum(ERIaa(C,D,1:NOccA,K)*T1a(:,C))*T1a(J,D)
           End Do
           Do C = NOccB+1, NBF
             Fkj(K,J) = Fkj(K,J) + Sum(ERIab(D,C,K,1:NOccB)*T1b(:,C))*T1a(J,D)
           End Do
           End Do
           Do L = 1, NOccA
             Fkj(K,J) = Fkj(K,J) + Sum(ERIaa(NOccA+1:NBF,J,L,K)*T1a(L,:))
           End Do
           Do L = 1, NOccB
             Fkj(K,J) = Fkj(K,J) + Sum(ERIab(J,NOccB+1:NBF,K,L)*T1b(L,:))
           end do
        End Do
        End Do
 
        Do J = 1, NOccA 
        Do I = 1, NOccA 
        Do A = NOccA+1, NBF 
        Do B = NOccA+1, NBF 
          G2aa(I,J,A,B) = G2aa(I,J,A,B) + Sum(Fkj(:,J)*T2aa(:,I,A,B))
        End Do
        End Do
        End Do
        End Do
!
! 
        Fcb = Zero
        Do C = NOccA+1, NBF
        Do B = NOccA+1, NBF
          Fcb(C,B) = Fcb(C,B) + Sum(FockA(1:NOccA,C)*T1a(:,B))
           Do L = 1, NOccA 
           Do D = NOccA+1, NBF
             Fcb(C,B) =  Fcb(C,B) + Sum(ERIaa(D,C,1:NOccA,L)*T1a(:,D))*T1a(L,B)
           End Do
           Do D = NOccB+1, NBF
             Fcb(C,B) =  Fcb(C,B) + Sum(ERIab(C,D,L,1:NOccB)*T1b(:,D))*T1a(L,B)
           End Do
           End Do
            do k = 1, nocca
              Fcb(C,B) =  Fcb(C,B) - Sum(ERIaa(NOcca+1:nbf,C,K,B)*T1a(K,:))
            end do
            do k = 1, noccb
              Fcb(C,B) =  Fcb(C,B) - Sum(ERIab(C,NOccB+1:NBF,B,K)*T1b(K,:))
            end do
        End Do
        End Do
! 
        Do I = 1, NOccA 
        Do J = 1, NOccA 
        Do A = NOccA+1, NBF 
        Do B = NOccA+1, NBF 
          G2aa(I,J,A,B) = G2aa(I,J,A,B) +  Sum(Fcb(:,B)*T2aa(I,J,:,A))
        End Do
        End Do
        End Do
        End Do
! !BetaBeta
        Fbil = Zero 
        Do I = 1, noccb
        Do L = 1, noccb
          Fbil(I,L) = Fbil(I,L) + Sum(fockb(L,noccb+1:NBF)*t1b(I,:))
           Do D = noccb+1, NBF
           Do K = 1, noccb
             Fbil(I,L) = Fbil(I,L) + Sum(eribb(noccb+1:NBF,D,K,L)*t1b(K,:))*t1b(I,D)
           End Do
           Do K = 1, nocca
             Fbil(I,L) = Fbil(I,L) + Sum(ERIab(nocca+1:NBF,D,K,L)*t1a(K,:))*t1b(I,D)
 
           End Do
           End Do
           Do K = 1, noccb
             Fbil(I,L) = Fbil(I,L) + Sum(eribb(noccb+1:NBF,I,K,L)*t1b(K,:))
           End Do
           Do K = 1, nocca
             Fbil(I,L) = Fbil(I,L) + Sum(ERIab(nocca+1:NBF,I,K,L)*t1a(K,:))
           End Do
        End Do
        End Do
        Fbil = -Fbil
! 
! 
        Do I = 1, NOccb 
        Do J = 1, NOccb 
        Do A = NOccb+1, NBF 
        Do B = NOccb+1, NBF 
          G2bb(I,J,A,B) = G2bb(I,J,A,B) + Sum(Fbil(I,:)*T2bb(:,J,A,B))
        End Do
        End Do
        End Do
        End Do
! 
        fbad = Zero
        Do A = NOccb+1, NBF
        Do D = NOccb+1, NBF
          fbad(A,D) = fbad(A,D) + Sum(Fockb(1:NOccb,D)*T1b(:,A))
           Do L = 1, NOccb
           Do K = 1, NOccb
             fbad(A,D) = fbad(A,D) + Sum(ERIbb(NOccb+1:NBF,D,K,L)*T1b(K,:))*T1b(L,A)
           End Do    
           Do K = 1, NOcca
             fbad(A,D) = fbad(A,D) + Sum(ERIab(NOcca+1:NBF,D,K,L)*T1a(K,:))*T1b(L,A)
           End Do
           End Do
           Do K = 1, NOccb
            fbad(A,D) = fbad(A,D) - Sum(ERIbb(NOccb+1:NBF,D,K,A)*T1b(K,:))
           End Do
           Do K = 1, NOcca
            fbad(a,d) = fbad(a,d) - sum(ERIab(NOcca+1:NBF,D,K,A)*T1a(K,:))
           End Do
        End Do
        End Do
        fbad = -fbad
 
        Do I = 1, NOccb 
        Do J = 1, NOccb 
        Do A = NOccb+1, NBF 
        Do B = NOccb+1, NBF 
          G2bb(I,J,A,B) = G2bb(I,J,A,B) + Sum(Fbad(A,:)*T2bb(I,J,:,B))
        End Do
        End Do
        End Do
        End Do
! 
! 
        fbkj = Zero
        Do K = 1, NOccb
        Do J = 1, NOccb
          fbkj(K,J) = fbkj(K,J) + Sum(Fockb(K,NOccb+1:NBF)*T1b(J,:))
           Do D = NOccb+1, NBF
           Do C = NOccb+1, NBF
             fbkj(K,J) = fbkj(K,J) + Sum(ERIbb(C,D,1:NOccb,K)*T1b(:,C))*T1b(J,D)
           End Do
           Do C = NOcca+1, NBF
             fbkj(K,J) = fbkj(K,J) + Sum(ERIab(C,D,1:NOcca,K)*T1a(:,C))*T1b(J,D)
           End Do
           End Do
           Do L = 1, NOccb
             fbkj(K,J) = fbkj(K,J) + Sum(ERIbb(NOccb+1:NBF,J,L,K)*T1b(L,:))
           End Do
           Do L = 1, NOcca
             fbkj(K,J) = fbkj(K,J) + Sum(ERIab(NOcca+1:NBF,J,L,K)*T1a(L,:))
           end do
        End Do
        End Do
! 
        Do J = 1, NOccb 
        Do I = 1, NOccb 
        Do A = NOccb+1, NBF 
        Do B = NOccb+1, NBF 
          G2bb(I,J,A,B) = G2bb(I,J,A,B) + Sum(Fbkj(:,J)*T2bb(:,I,A,B))
        End Do
        End Do
        End Do
        End Do
 
 
       fbcb = Zero
       Do C = NOccb+1, NBF
       Do B = NOccb+1, NBF
         fbcb(C,B) = fbcb(C,B) + Sum(Fockb(1:NOccb,C)*T1b(:,B))
          Do L = 1, NOccb 
          Do D = NOccb+1, NBF
            fbcb(C,B) =  fbcb(C,B) + Sum(ERIbb(D,C,1:NOccb,L)*T1b(:,D))*T1b(L,B)
          End Do
          Do D = NOcca+1, NBF
            fbcb(C,B) =  fbcb(C,B) + Sum(ERIab(D,C,1:NOcca,L)*T1a(:,D))*T1b(L,B)
          End Do
          End Do
           do k = 1, noccb
             fbcb(C,B) =  fbcb(C,B) - Sum(ERIbb(NOccb+1:nbf,C,K,B)*T1b(K,:))
           end do
           do k = 1, nocca
             fbcb(C,B) =  fbcb(C,B) - Sum(ERIab(NOcca+1:NBF,C,K,B)*T1a(K,:))
           end do
       End Do
       End Do
! 
       Do I = 1, NOccb 
       Do J = 1, NOccb 
       Do A = NOccb+1, NBF 
       Do B = NOccb+1, NBF 
         G2bb(I,J,A,B) = G2bb(I,J,A,B) +  Sum(Fbcb(:,B)*T2bb(I,J,:,A))
       End Do
       End Do
       End Do
       End Do
!!
!!!alpha/beta 
!!!We will re-use intermediates as possible
!!
!!!Fil is identical
!!!Fad is identical
!!!Fbcb and Fbkj swap signs
        Do I = 1, NOccA 
        Do J = 1, NOccb 
        Do A = NOccA+1, NBF 
        Do B = NOccb+1, NBF 
          G2ab(I,J,A,B) = G2ab(I,J,A,B) + Sum(Fil(I,:)*T2ab(:,J,A,B))&
                                        + Sum(Fad(A,:)*T2ab(I,J,:,B))&
                                        - Sum(Fbkj(:,J)*T2ab(I,:,A,B))&
                                        - Sum(Fbcb(:,B)*T2ab(I,J,A,:))
        End Do
        End Do
        End Do
        End Do
!
!
!
!!!
!!!
!!!print *, "**************************************************"
!!!        Print *, "CCSD aa Double after F : ", G2aa
!!!        Print *, "CCSD bb Double after F : ", G2bb
!!!        Print *, "CCSD ab Double after F : ", G2ab
!!!print *, "**************************************************"
!!!
!!!
!!
!!!Make and use W intermediates
!!!alpha
        wilad = Zero
        wikdb = Zero
        wljac = Zero
        wkjdb = Zero
        wcdab = Zero
        wijkl = Zero

        Do I = 1, NOccA
        Do L = 1, NOccA
        Do A = NOccA+1, nbf
        Do D = NOccA+1, nbf

         wilad(i,l,a,d) = wilad(i,l,a,d) + Sum(ERIaa(nocca+1:nbf,D,A,L)*T1a(I,:))&
                                          - Sum(ERIaa(I,D,1:nocca,L)*T1a(:,A))
          Do k = 1, nocca
            wilad(i,l,a,d) = wilad(i,l,a,d) - sum(ERIaa(nocca+1:nbf,D,K,L)*T1a(I,:))*T1a(K,A)
          end do
        End Do
        End Do
        End Do
        End Do
!!!    
!!!
        Do I = 1, NOccA
        Do k = 1, NOccA
        Do d = NOccA+1, nbf
        Do b = NOccA+1, nbf
          wikdb(i,k,d,b) = wikdb(i,k,d,b) - sum(ERIaa(nocca+1:nbf,D,B,K)*T1a(I,:))&
                                          + sum(ERIaa(I,d,1:nocca,k)*T1a(:,B))

          do l = 1, nocca
          wikdb(i,k,d,b) = wikdb(i,k,d,b) + sum(ERIaa(nocca+1:nbf,D,l,k)*T1a(I,:))*T1a(l,B)
          end do
        End Do
        End Do
        End Do
        End Do

        Do l = 1, NOccA
        Do j = 1, NOccA
        Do a = NOccA+1, nbf
        Do c = NOccA+1, nbf
          wljac(l,j,a,c) = wljac(l,j,a,c) - sum(ERIaa(nocca+1:nbf,c,A,l)*T1a(J,:))&
                                          + sum(ERIaa(J,C,1:nocca,L)*T1a(:,A))
          do k = 1, nocca
          wljac(l,j,a,c) = wljac(l,j,a,c) + sum(ERIaa(nocca+1:nbf,c,K,L)*T1a(J,:))*T1a(K,A)
          end do

       End Do
       End Do
       End Do
       End Do

       Do k = 1, NOccA
       Do j = 1, NOccA
       Do d = NOccA+1, nbf
       Do b = NOccA+1, nbf
          wkjdb(k,j,d,b) = wkjdb(k,j,d,b) + sum(ERIaa(nocca+1:nbf,D,B,K)*T1a(J,:))&
                                          - sum(ERIaa(J,d,1:nocca,k)*T1a(:,B))
          do l = 1, nocca
          wkjdb(k,j,d,b) = wkjdb(k,j,d,b) - sum(ERIaa(nocca+1:nbf,D,l,k)*T1a(J,:))*T1a(l,B)
          end do
       End Do
       End Do
       End Do
       End Do

        do c = nocca+1, nbf
        do d = nocca+1, nbf
        do a = nocca+1, nbf
        do b = nocca+1, nbf
          wcdab(c,d,a,b) = wcdab(c,d,a,b) - sum(ERIaa(C,D,1:nocca,B)*T1a(:,A))&
                                      + sum(ERIaa(C,D,1:nocca,A)*T1a(:,B))
          do l = 1, nocca
            wcdab(c,d,a,b) = wcdab(c,d,a,b) + sum(ERIaa(C,D,1:nocca,L)*T1a(:,A))*T1a(L,B)
          end do
        end do
        end do
        end do
        end do
        wcdab = F12*wcdab
!!!!
        do i = 1, nocca
        do j = 1, nocca
        do k = 1, nocca
        do l = 1, nocca
          wijkl(i,j,k,l) = wijkl(i,j,k,l) + sum(ERIaa(NOccA+1:NBF,J,K,L)*T1a(I,:))&
                                      - sum(ERIaa(nocca+1:nbf,I,K,L)*T1a(J,:))
          do d = nocca + 1, nbf
            wijkl(i,j,k,l) = wijkl(i,j,k,l) + sum(ERIaa(nocca+1:nbf,D,K,L)*T1a(I,:))*T1a(J,D)
          end do
        end do
        end do
        end do
        end do
        wijkl = F12*wijkl
!!!!
!!!!
!!!!
        Do I = 1, NOccA 
        Do J = 1, NOccA 
        Do A = NOccA+1, NBF 
        Do B = NOccA+1, NBF 
          do d = nocca+1, nbf
            G2aa(I,J,A,B) = G2aa(I,J,A,B) + Sum(wilad(i,:,a,d)*T2aa(:,j,d,b))&
                                          + sum(wikdb(i,:,d,b)*T2aa(:,J,D,A))&
                                          + sum(wljac(:,j,a,d)*T2aa(:,I,D,B))&
                                          + sum(wkjdb(:,j,d,b)*T2aa(:,I,D,A))&
                                          + sum(wcdab(:,d,a,b)*T2aa(I,J,:,D))
         end do
         do l = 1, nocca
            G2aa(I,J,A,B) = G2aa(I,J,A,B) + sum(wijkl(i,j,:,l)*T2aa(:,l,A,B))
          end do
        End Do
        End Do
        End Do
        End Do
!!!
!!!!beta/beta
!!!!beta
       wbilad = Zero 
       wbikdb = Zero 
       wbljac = Zero 
       wbkjdb = Zero 
       wbcdab = Zero 
       wbijkl = Zero 
!
        Do i = 1, noccb
        Do l = 1, noccb
        Do a = noccb+1, nbf
        Do d = noccb+1, nbf
          wbilad(i,l,a,d) = wbilad(i,l,a,d) + sum(ERIbb(noccb+1:nbf,D,A,l)*T1b(I,:))&
                                            - sum(ERIbb(I,d,1:noccb,L)*T1b(:,A))
          do k = 1, noccb
            wbilad(i,l,a,d) = wbilad(i,l,a,d) - sum(ERIbb(noccb+1:nbf,D,K,L)*T1b(I,:))*T1b(K,A)
          end do 
        end do
        end do
        end do
        end do

        Do i = 1, noccb
        Do k = 1, noccb
        Do d = noccb+1, nbf
        Do b = noccb+1, nbf
          wbikdb(i,k,d,b) = wbikdb(i,k,d,b) - sum(ERIbb(noccb+1:nbf,D,B,K)*T1b(I,:))&
                                            + sum(ERIbb(I,d,1:noccb,k)*T1b(:,B))
          do l = 1, noccb
            wbikdb(i,k,d,b) = wbikdb(i,k,d,b) + sum(ERIbb(noccb+1:nbf,D,L,K)*T1b(I,:))*T1b(L,B)
          end do 
        end do
        end do
        end do
        end do

        Do l = 1, noccb
        Do j = 1, noccb
        Do a = noccb+1, nbf
        Do c = noccb+1, nbf
          wbljac(l,j,a,c) = wbljac(l,j,a,c) - sum(ERIbb(noccb+1:nbf,c,A,l)*T1b(J,:))&
                                            + sum(ERIbb(J,C,1:noccb,L)*T1b(:,A))
          do k = 1, noccb
            wbljac(l,j,a,c) = wbljac(l,j,a,c) + sum(ERIbb(noccb+1:nbf,c,K,L)*T1b(J,:))*T1b(K,A)
          end do 
        end do
        end do
        end do
        end do


        Do k = 1, NOccb
        Do j = 1, NOccb
        Do d = NOccb+1, nbf
        Do b = NOccb+1, nbf
          wbkjdb(k,j,d,b) = wbkjdb(k,j,d,b) + sum(ERIbb(noccb+1:nbf,D,B,K)*T1b(J,:))&
                                          - sum(ERIbb(J,d,1:noccb,k)*T1b(:,B))
          do l = 1, noccb
          wbkjdb(k,j,d,b) = wbkjdb(k,j,d,b) - sum(ERIbb(noccb+1:nbf,D,l,k)*T1b(J,:))*T1b(l,B)
          end do
        End Do
        End Do
        End Do
        End Do

!!!
       do c = noccb+1, nbf
       do d = noccb+1, nbf
       do a = noccb+1, nbf
       do b = noccb+1, nbf
          wbcdab(c,d,a,b) = wbcdab(c,d,a,b) - sum(ERIbb(C,D,1:noccb,B)*T1b(:,A))&
                                      + sum(ERIbb(C,D,1:noccb,A)*T1b(:,B))
         do l = 1, noccb
           wbcdab(c,d,a,b) = wbcdab(c,d,a,b) + sum(ERIbb(C,D,1:noccb,L)*T1b(:,A))*T1b(L,B)
         end do
       end do
       end do
       end do
       end do
       wbcdab = F12*wbcdab
!!!
!!!
       do i = 1, noccb
       do j = 1, noccb
       do k = 1, noccb
       do l = 1, noccb
          wbijkl(i,j,k,l) = wbijkl(i,j,k,l) + sum(ERIbb(NOccb+1:NBF,J,K,L)*T1b(I,:))&
                                      - sum(ERIbb(noccb+1:nbf,I,K,L)*T1b(J,:))
         do d = noccb + 1, nbf
           wbijkl(i,j,k,l) = wbijkl(i,j,k,l) + sum(ERIbb(noccb+1:nbf,D,K,L)*T1b(I,:))*T1b(J,D)
         end do
       end do
       end do
       end do
       end do
       wbijkl = F12*wbijkl
!!
!!!
!!!
       Do I = 1, NOccb 
       Do J = 1, NOccb 
       Do A = NOccb+1, NBF 
       Do B = NOccb+1, NBF 
        do d = noccb+1, nbf
          G2bb(I,J,A,B) = G2bb(I,J,A,B)  + Sum(wbilad(i,:,a,d)*T2bb(:,j,d,b))&
                                         + sum(wbikdb(i,:,d,b)*T2bb(:,J,D,A))&
                                         + sum(wbljac(:,j,a,d)*T2bb(:,I,D,B))&
                                         + sum(wbkjdb(:,j,d,b)*T2bb(:,I,D,A))&
                                         + sum(wbcdab(:,d,a,b)*T2bb(I,J,:,D))
         end do
         do l = 1, noccb
          G2bb(I,J,A,B) = G2bb(I,J,A,B)  + sum(wbijkl(i,j,:,l)*T2bb(:,l,A,B))
         end do
       End Do
       End Do
       End Do
       End Do
!!
!!!alpha/beta
!!
!!!wilad is identical
!!!wb kjdb is identical
        do i = 1, nocca
        do j = 1, noccb
        do a = nocca+1, nbf
        do b = noccb+1, nbf
          do d = nocca+1, nbf
            g2ab(i,j,a,b) = g2ab(i,j,a,b) + sum(wilad(i,:,a,d)*T2ab(:,j,d,b))
          end do
          do d = noccb+1, nbf
            g2ab(i,j,a,b) = g2ab(i,j,a,b) + sum(wbkjdb(:,j,d,b)*t2ab(I,:,A,D))
          end do
        end do
        end do
        end do
        end do


!!!
!!!print *, "**************************************************"
!!!        Print *, "CCSD aa Double after W : ", G2aa
!!!        Print *, "CCSD bb Double after W : ", G2bb
!!!        Print *, "CCSD ab Double after W : ", G2ab
!!!print *, "**************************************************"
!!!
!!!
!!
!!!Make and use M intermediates
!!!alpha/alpha
        mikad = Zero
        mikbd = Zero
        mjkad = Zero
        mjlbc = Zero
        do i = 1, nocca
        do k = 1, noccb
        do a = nocca+1, nbf
        do d = noccb+1, nbf
          mikad(i,k,a,d) = mikad(i,k,a,d) + sum(ERIab(nocca+1:nbf,D,A,K)*T1a(I,:))&
                                          - sum(ERIab(I,D,1:nocca,K)*T1a(:,A))
          do l = 1, nocca
            mikad(i,k,a,d) = mikad(i,k,a,d) - sum(ERIab(nocca+1:nbf,D,l,k)*T1a(I,:))*T1a(l,A)
          end do

        end do
        end do
        end do
        end do

        do i = 1, nocca
        do k = 1, noccb
        do b = nocca+1, nbf
        do d = noccb+1, nbf
          mikbd(i,k,b,d) = mikbd(i,k,b,d) - sum(ERIab(nocca+1:nbf,D,B,K)*T1a(I,:))&
                                          + sum(ERIab(I,d,1:nocca,k)*T1a(:,B))
          do l = 1, nocca
            mikbd(i,k,b,d) = mikbd(i,k,b,d) + sum(ERIab(nocca+1:nbf,D,l,k)*T1a(I,:))*T1a(l,B)
          end do

        end do
        end do
        end do
        end do

        do j = 1, nocca
        do k = 1, noccb
        do a = nocca+1, nbf
        do d = noccb+1, nbf
          mjkad(j,k,a,d) = mjkad(j,k,a,d) - sum(ERIab(nocca+1:nbf,D,A,K)*T1a(J,:))&
                                          + sum(ERIab(J,d,1:nocca,k)*T1a(:,A))
!!
          do l = 1, nocca
            mjkad(j,k,a,d) = mjkad(j,k,a,d) + sum(ERIab(nocca+1:nbf,D,l,k)*T1a(J,:))*T1a(l,A)
          end do

        end do
        end do
        end do
        end do



        do j = 1, nocca
        do l = 1, noccb
        do b = nocca+1, nbf
        do c = noccb+1, nbf
          mjlbc(j,l,b,c) = mjlbc(j,l,b,c) + sum(ERIab(nocca+1:nbf,c,B,l)*T1a(J,:))&
                                          - sum(ERIab(J,C,1:nocca,L)*T1a(:,B))
          do k = 1, nocca
            mjlbc(j,l,b,c) = mjlbc(j,l,b,c) - sum(ERIab(nocca+1:nbf,c,K,L)*T1a(J,:))*T1a(K,B)
          end do

        end do
        end do
        end do
        end do


        Do I = 1, NOccA 
        Do J = 1, NOccA 
        Do A = NOccA+1, NBF 
        Do B = NOccA+1, NBF 
          do d = noccb+1, nbf
            G2aa(I,J,A,B) = G2aa(I,J,A,B)  + sum(mikad(i,1:noccb,a,d)*T2ab(J,:,B,D))&
                                           + sum(mikbd(i,1:noccb,b,d)*T2ab(J,:,A,D))&
                                           + sum(mjkad(j,1:noccb,a,d)*T2ab(I,:,B,D))&
                                           + sum(mjlbc(j,1:noccb,b,d)*T2ab(I,:,A,D))

          end do
        End Do
        End Do
        End Do
        End Do
      
!beta/beta
        mbikad = Zero
        mbikbd = Zero
        mbjkad = Zero
        mbjlbc = Zero

        do i = 1, noccb
        do k = 1, nocca
        do a = noccb+1, nbf
        do d = nocca+1, nbf
          mbikad(i,k,a,d) = mbikad(i,k,a,d) + sum(ERIab(D,noccb+1:nbf,K,A)*t1b(I,:))&
                                          - sum(ERIab(D,I,K,1:noccb)*t1b(:,A))
          do l = 1, noccb
            mbikad(i,k,a,d) = mbikad(i,k,a,d) - sum(ERIab(D,noccb+1:nbf,k,l)*t1b(I,:))*t1b(l,A)
          end do

        end do
        end do
        end do
        end do

        do i = 1, noccb
        do k = 1, nocca
        do b = noccb+1, nbf
        do d = nocca+1, nbf
          mbikbd(i,k,b,d) = mbikbd(i,k,b,d) - sum(ERIab(d,noccb+1:nbf,K,b)*t1b(I,:))&
                                          + sum(ERIab(d,i,k,1:noccb)*t1b(:,B))
          do l = 1, noccb
            mbikbd(i,k,b,d) = mbikbd(i,k,b,d) + sum(ERIab(d,noccb+1:nbf,k,l)*t1b(I,:))*t1b(l,B)
          end do

        end do
        end do
        end do
        end do

        do j = 1, noccb
        do k = 1, nocca
        do a = noccb+1, nbf
        do d = nocca+1, nbf
          mbjkad(j,k,a,d) = mbjkad(j,k,a,d) - sum(ERIab(d,noccb+1:nbf,K,a)*t1b(J,:))&
                                          + sum(ERIab(d,j,k,1:noccb)*t1b(:,A))
!!
          do l = 1, noccb
            mbjkad(j,k,a,d) = mbjkad(j,k,a,d) + sum(ERIab(d,noccb+1:nbf,k,l)*t1b(J,:))*t1b(l,A)
          end do

        end do
        end do
        end do
        end do



        do j = 1, noccb
        do l = 1, nocca
        do b = noccb+1, nbf
        do c = nocca+1, nbf
          mbjlbc(j,l,b,c) = mbjlbc(j,l,b,c) + sum(ERIab(c,noccb+1:nbf,l,B)*t1b(J,:))&
                                          - sum(ERIab(c,j,l,1:noccb)*t1b(:,B))
          do k = 1, noccb
            mbjlbc(j,l,b,c) = mbjlbc(j,l,b,c) - sum(ERIab(c,noccb+1:nbf,l,k)*t1b(J,:))*t1b(K,B)
          end do

        end do
        end do
        end do
        end do


        Do I = 1, noccb 
        Do J = 1, noccb 
        Do A = noccb+1, NBF 
        Do B = noccb+1, NBF 
          do d = nocca+1, nbf
            G2bb(I,J,A,B) = G2bb(I,J,A,B)  + sum(mbikad(i,1:nocca,a,d)*T2ab(:,J,D,B))&
                                           + sum(mbikbd(i,1:nocca,b,d)*T2ab(:,J,D,A))&
                                           + sum(mbjkad(j,1:nocca,a,d)*T2ab(:,I,D,B))&
                                           + sum(mbjlbc(j,1:nocca,b,d)*T2ab(:,I,D,A))

          end do
        End Do
        End Do
        End Do
        End Do
      


!alpha/beta
!as far as I can tell, only mikad and mbljcb are re-used. There may be other repeats, but at
!this point it will just be quicker to code new intermediates
       Do i = 1, nocca
       Do j = 1, noccb
       do a = nocca+1, nbf
       do b = noccb+1, nbf
         do d = noccb+1, nbf
           g2ab(i,j,a,b) = g2ab(i,j,a,b) + sum(mikad(i,:,a,d)*t2bb(:,J,D,B))
         end do
         do c = nocca+1, nbf
           g2ab(i,j,a,b) = g2ab(i,j,a,b) + sum(mbjlbc(j,:,b,c)*T2aa(:,I,c,A))
         end do

       end do
       end do
       end do
       end do


!
!print *, "**************************************************"
!        Print *, "CCSD aa Double after M : ", G2aa
!        Print *, "CCSD bb Double after M : ", G2bb
!        Print *, "CCSD ab Double after M : ", G2ab
!print *, "**************************************************"
!

!Now we will do the alpha beta terms that have no corresponding intermediates in
!the alpha/alpha, alpha/beta equations


      Kjkad = Zero
      Kildb = Zero
      !Rljdb = Zero
      Kklij = Zero
      Kcdab = Zero

      do j = 1, noccb
      do k = 1, noccb
      do a = nocca+1, nbf
      do d = nocca+1, nbf
        Kjkad(j,k,a,d) = Kjkad(j,k,a,d) - sum(ERIab(D,noccb+1:nbf,A,K)*t1b(J,:))&
                                        + sum(ERIab(d,J,1:nocca,k)*t1a(:,A))
        do l = 1, nocca
          Kjkad(j,k,a,d) = Kjkad(j,k,a,d) + sum(ERIab(D,noccb+1:nbf,l,k)*t1b(J,:))*t1a(l,A)
        end do

      end do
      end do
      end do
      end do

      do i = 1, nocca
      do l = 1, nocca
      do d = noccb+1, nbf
      do b = noccb+1, nbf
        do k = 1, noccb
          Kildb(i,l,d,b) = Kildb(i,l,d,b) + sum(ERIab(nocca+1:nbf,D,L,K)*t1a(I,:))*t1b(K,B)
        end do
          Kildb(i,l,d,b) = Kildb(i,l,d,b) - sum(ERIab(nocca+1:nbf,D,l,B)*t1a(I,:))&
                                          + sum(ERIab(I,d,L,1:noccb)*t1b(:,B))
      end do
      end do
      end do
      end do


!       do l = 1, nocca
!       do j = 1, noccb
!       do d = nocca+1, nbf
!       do b = noccb+1, nbf
!!!         do k = 1, noccb
!!!           Rljdb(l,j,d,b) = Rljdb(l,j,d,b) - sum(ERIab(D,noccb+1:nbf,L,K)*t1b(J,:))*t1b(K,B)
!!!         end do
!           Rljdb(l,j,d,b) = Rljdb(l,j,d,b) &!+ sum(ERIab(D,noccb+1:nbf,l,B)*t1b(J,:))&
!                                           - sum(ERIab(d,J,L,1:noccb)*t1b(:,B))
!       end do
!       end do
!       end do
!       end do
!!
       do k = 1, nocca
       do l = 1, noccb
       do i = 1, nocca
       do j = 1, noccb
          Kklij(k,l,i,j) = Kklij(k,l,i,j) + sum(ERIab(I,noccb+1:nbf,K,L)*t1b(J,:))&
                                          + sum(ERIab(nocca+1:nbf,J,K,L)*T1a(I,:))
         do d = noccb+1, nbf
           Kklij(k,l,i,j) = Kklij(k,l,i,j) + sum(ERIab(nocca+1:nbf,D,k,l)*t1a(I,:))*t1b(J,D)
         end do
       end do
       end do
       end do
       end do
!!
       do c = nocca+1, nbf
       do d = noccb+1, nbf
       do a = nocca+1, nbf
       do b = noccb+1, nbf
          Kcdab(c,d,a,b) = Kcdab(c,d,a,b) - sum(ERIab(C,D,A,1:noccb)*t1b(:,B))&
                                          - sum(ERIab(C,D,1:nocca,B)*t1a(:,A))
         do l = 1, noccb
           Kcdab(c,d,a,b) = Kcdab(c,d,a,b) + sum(ERIab(C,D,1:nocca,L)*t1a(:,A))*t1b(L,B)
         end do
       end do
       end do
       end do
       end do
!        
!!
!!
!
       Do i = 1, nocca
       Do j = 1, noccb
       do a = nocca+1, nbf
       do b = noccb+1, nbf
         do d = nocca+1, nbf
           g2ab(i,j,a,b) = g2ab(i,j,a,b) + sum(Kjkad(j,:,a,d)*t2ab(I,:,D,B))
         end do
         do d = noccb+1, nbf
           g2ab(i,j,a,b) = g2ab(i,j,a,b) + sum(Kildb(i,:,d,b)*t2ab(:,J,A,D))
         end do
!!       !  do d = nocca+1, nbf
!!       !    g2ab(i,j,a,b) = g2ab(i,j,a,b) + sum(Rljdb(:,j,d,b)*t2aa(:,I,D,A))
!!       !  end do
        do l = 1, noccb
          g2ab(i,j,a,b) = g2ab(i,j,a,b) + sum(Kklij(:,l,i,j)*t2ab(:,L,A,B))
        end do
        do d = noccb+1, nbf
          g2ab(i,j,a,b) = g2ab(i,j,a,b) + sum(Kcdab(:,d,a,b)*t2ab(I,J,:,D))
        end do
      end do
      end do
      end do
      end do

       deallocate(Fil,&
               Fad,&
               Fkj,&
               Fcb,&
               wilad,&
               wikdb,&
               wljac,&
               wkjdb,&
               wcdab,&
               wijkl,&
               mikad,&
               mikbd,&
               mjkad,&
               mjlbc,&
               Fbil,&
               Fbad,&
               Fbkj,&
               Fbcb,&
               wbijkl,&
               wbilad,&
               wbikdb,&
               wbljac,&
               wbkjdb,&
               wbcdab,&
               mbikad,&
               mbikbd,&
               mbjkad,&
               mbjlbc,&
               Kjkad,    &
               Kildb,    &
               Rljdb,    &
               Kcdab,    &
               Kklij)


      End Subroutine GetCCSDG2



      Subroutine CCSDEnergy(T1a,T1b,T2aa,T2ab,T2bb,FockA,FockB,ERIaa,ERIbb,ERIab,oA,oB,vA,vB,NBF,ECorr)
      !=====================================================================!
      ! This subroutine gets the CCD correlation energy within a UHF-style framework. 
      ! Suitable for molecules.
      ! -jag20, 7.29.15
      !=====================================================================!
      Implicit None
      Integer,        Intent(In)  :: oA, oB, vA, vB, NBF
      double precision, Intent(In)  :: FockA(NBF,NBF)
      double precision, Intent(In)  :: FockB(NBF,NBF)
      double precision, Intent(In)  :: ERIaa(NBF,NBF,NBF,NBF),&
                                      ERIab(NBF,NBF,NBF,NBF),&
                                      ERIbb(NBF,NBF,NBF,NBF)
      double precision, Intent(In) :: &
                                      T1a(1:oA,vA:NBF),&
                                      T1b(1:oB,vB:NBF)
      double precision, Intent(In)  :: &
                                      T2aa(1:oA,1:oA,vA:NBF,vA:NBF),&
                                      T2ab(1:oA,1:oB,vA:NBF,vB:NBF),&
                                      T2bb(1:oB,1:oB,vB:NBF,vB:NBF)

      !f2py depend(NBF) FockA, FockB, EriAA, EriAB, ERIbb, T2aa, T2ab,T2bb, T1a, T1b
      !f2py depend(NoccA) T2aa, T2ab, T1a
      !f2py depend(NoccB) T2ab, T2bb, T1b
      double precision, Intent(Out) :: ECorr
      double precision              :: ECorr1, ECCD, ECorr2, ECorrAA, ECorrBB, ECorrAB
      Integer                     :: I, J, A, B
      double precision::                   Zero =0.0e0, F12 = 0.50e0, F14 = 0.25e0

!     CCD Energy Contributions
!     Get T2aa energy. We include this T2 term in ECorrAA.
      ECorrAA = Zero
      Do I = 1, oA
      Do J = 1, oA
      Do A = vA, NBF
      Do B = vA, NBF
        ECorrAA = ECorrAA + F14*(ERIaa(A,B,I,J)*T2aa(I,J,A,B))
      End Do
      End Do
      End Do
      End Do

!     Get T2ab energy. We include this T2 term in ECorrAB.
      ECorrAB = Zero
      Do I = 1, oA
      Do J = 1, oB
      Do A = vA, NBF
      Do B = vB, NBF
        ECorrAB = ECorrAB + ERIab(A,B,I,J)*T2ab(I,J,A,B)
      End Do
      End Do
      End Do
      End Do

!     Get T2bb energy. We include this T2 term in ECorrBB.
      ECorrBB = Zero
      Do I = 1, oB
      Do J = 1, oB
      Do A = vB, NBF
      Do B = vB, NBF
        ECorrBB = ECorrBB + F14*(ERIbb(A,B,I,J)*T2bb(I,J,A,B))
      End Do
      End Do
      End Do
      End Do
      ECCD = ECorrAA + ECorrAB + ECorrBB

! CCSD Energy terms involving Single Excitations
!     Linear Terms
      ECorr1 = Zero
      do i = 1, oA
      do a = vA, nbf
        ECorr1 = ECorr1 + FockA(I,A)*T1a(I,A)
      end do
      end do

      do i = 1, oB
      do a = vB, nbf
        ECorr1 =ECorr1 + FockB(I,A)*T1b(I,A)
      end do
      end do
!     Quadratic Terms
      ECorr2 = Zero
      Do I = 1, oA
      Do J = 1, oA
      Do A = vA, NBF
      Do B = vA, NBF
      ECorr2 =ECorr2 + F12*ERIaa(A,B,I,J)*T1a(I,A)*T1a(J,B)
      End Do
      End Do
      End Do
      End Do

      Do I = 1, oA
      Do J = 1, oB
      Do A = vA, NBF
      Do B = vB, NBF
      ECorr2 =ECorr2 + F12*ERIab(A,B,I,J)*T1a(I,A)*T1b(J,B)
      end do
      end do
      end do
      end do

      Do j = 1, oA
      Do i = 1, oB
      Do b = vA, NBF
      Do a = vB, NBF
      ECorr2 =ECorr2 + F12*ERIab(B,A,J,I)*T1b(I,A)*T1a(J,B)
      end do
      end do
      end do
      end do

      Do i = 1, oB
      Do j = 1, oB
      Do a = vB, NBF
      Do b = vB, NBF
      ECorr2 =ECorr2 + F12*ERIbb(A,B,I,J)*T1b(I,A)*T1b(J,B)
      end do
      end do
      end do
      end do

! Output the total correlation energy
      ECorr  = ECCD + ECorr1 + ECorr2
      End Subroutine CCSDEnergy



