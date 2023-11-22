! This performs the calculations necessary to run a simulation with the Aziz (1979) potential
! 
! Copyright (C) 2013, Joshua More and Michele Ceriotti
! 
! Permission is hereby granted, free of charge, to any person obtaining
! a copy of this software and associated documentation files (the
! "Software"), to deal in the Software without restriction, including
! without limitation the rights to use, copy, modify, merge, publish,
! distribute, sublicense, and/or sell copies of the Software, and to
! permit persons to whom the Software is furnished to do so, subject to
! the following conditions:
! 
! The above copyright notice and this permission notice shall be included
! in all copies or substantial portions of the Software.
! 
! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
! EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
! MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
! IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
! CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
! TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
! SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
!
!
! This contains the functions that calculate the potential and forces of
! a single-component Aziz system.
! Includes functions which calculate the long-range correction terms for a
! simulation with a sharp nearest-neighbour cut-off.
!
! Functions:
!    aziz_functions: Calculates the Aziz pair potential and the magnitude of the
!       forces acting on a pair of atoms.
!    aziz_fij: Calculates the Aziz pair potential and force vector for the
!       interaction of a pair of atoms.
!    aziz_longrange: Calculates the long range correction to the potential
!       and virial.
!    aziz_getall: Calculates the potential of the system and the
!       forces acting on all the atoms.

      MODULE Aziz
         USE DISTANCE
      IMPLICIT NONE

      DOUBLE PRECISION, PARAMETER :: a_alpha = 13.353384
      DOUBLE PRECISION, PARAMETER :: a_A = 544850.4
      DOUBLE PRECISION, PARAMETER :: a_eps = 0.0000342 ! In Hartrees
      DOUBLE PRECISION, PARAMETER :: a_C6 = 1.3732412
      DOUBLE PRECISION, PARAMETER :: a_C8 = 0.4253785
      DOUBLE PRECISION, PARAMETER :: a_C10 = 0.178100
      DOUBLE PRECISION, PARAMETER :: a_D = 1.241314
      DOUBLE PRECISION, PARAMETER :: a_rm = 5.607 ! In Bohr radii

      CONTAINS

         SUBROUTINE aziz_functions(r, pot, force)
            ! Calculates the magnitude of the Aziz force and potential between
            ! a pair of atoms at a given distance from each other.
            !
            ! Args:
            !    r: The separation of the atoms.
            !    pot: The Aziz interaction potential.
            !    force: The magnitude of the Aziz force.

            DOUBLE PRECISION, INTENT(IN) :: r
            DOUBLE PRECISION, INTENT(OUT) :: pot
            DOUBLE PRECISION, INTENT(OUT) :: force
            
            DOUBLE PRECISION :: f, ratio, exp_term, sum_term
            DOUBLE PRECISION :: df, f_exp_term, f_sum1, f_sum2
            
            ! Calculating the potential and the force
            f = 1.0D0
            df = 0.0D0
            
            IF (r / a_rm <= a_D) THEN
                f = EXP(-(a_D * a_rm / r - 1.0D0)**2)
                df = (2.0D0 * a_D * a_rm * (a_D * a_rm - r)) * EXP(-(a_D * a_rm / r - 1.0D0) ** 2) / r ** 3
            END IF
            
            ratio = a_rm / r

            exp_term = EXP(-a_alpha / ratio)
            sum_term = (a_C6 * (ratio ** 6) + a_C8 * (ratio ** 8) + a_C10 * (ratio ** 10)) * f

            pot = a_eps * (a_A * exp_term - sum_term)
            
            ! Force evaluation
            f_exp_term = a_alpha * a_A * EXP(-a_alpha / ratio) / a_rm
            f_sum1 = (a_C6 * (ratio ** 6) + a_C8 * (ratio ** 8) + a_C10 * (ratio ** 10)) * df
            f_sum2 = (6.0D0 * a_C6 * (ratio ** 7) + 8.0D0 * a_C8 * (ratio ** 9) + 10.0D0 * a_C10 * (ratio ** 11)) * (f / a_rm)
            
            force = a_eps * (f_exp_term + f_sum1 - f_sum2)

         END SUBROUTINE

         SUBROUTINE aziz_fij(rij, r, pot, fij)
            ! This calculates the Aziz potential energy and the magnitude and
            ! direction of the force acting on a pair of atoms.
            !
            ! Args:
            !    sigma: The Aiz distance parameter.
            !    eps: The Aziz energy parameter.
            !    rij: The vector joining the two atoms.
            !    r: The separation of the two atoms.
            !    pot: The Aziz interaction potential.
            !    fij: The Aziz force vector.

            DOUBLE PRECISION, DIMENSION(3), INTENT(IN) :: rij
            DOUBLE PRECISION, INTENT(IN) :: r
            DOUBLE PRECISION, INTENT(OUT) :: pot
            DOUBLE PRECISION, DIMENSION(3), INTENT(OUT) :: fij
   
            DOUBLE PRECISION f_tot
   
            CALL aziz_functions(r, pot, f_tot)
            fij = f_tot*rij/r
   
         END SUBROUTINE

         SUBROUTINE aziz_longrange(rc, natoms, volume, pot_lr, vir_lr)
            ! Calculates the long range correction to the total potential and
            ! virial pressure.
            !
            ! Uses the tail correction for a sharp cut-off, with no smoothing
            ! function, as derived in Martyna and Hughes, Journal of Chemical
            ! Physics, 110, 3275, (1999).
            !
            ! Args:
            !    rc: The cut-off radius.
            !    natoms: The number of atoms in the system.
            !    volume: The volume of the system box.
            !    pot_lr: The tail correction to the Aziz interaction potential.
            !    vir_lr: The tail correction to the Aziz virial pressure.

            DOUBLE PRECISION, INTENT(IN) :: rc
            INTEGER, INTENT(IN) :: natoms
            DOUBLE PRECISION, INTENT(IN) :: volume
            DOUBLE PRECISION, INTENT(OUT) :: pot_lr
            DOUBLE PRECISION, INTENT(OUT) :: vir_lr

            DOUBLE PRECISION maxsep, rmol, rm3, t1, t2, t3, t4

            ! Del Maestro's implementation
            ! Check whether maxsep is indeed chosen correctly
            ! Also check where the cutoff enters the picture
            maxsep = volume**(1d0/3d0)

            rmol = a_rm / maxsep
            rm3 = a_rm * a_rm * rmol
            t1 = a_A * EXP(-a_alpha * maxsep / (2.0D0 * a_rm)) * a_rm * (8.0D0 * a_rm * a_rm + 4.0D0 * maxsep * a_rm * a_alpha + maxsep * maxsep * a_alpha * a_alpha) / (4.0D0 * a_alpha * a_alpha * a_alpha)
            t2 = 8.0D0 * a_C6 * rmoL**3.0D0 / 3.0D0
            t3 = 32.0D0 * a_C8 * rmoL**5.0D0 / 5.0D0
            t4 = 128.0D0 * a_C10 * rmoL**7.0D0 / 7.0D0

            pot_lr = 2.0D0 * ACOS(-1.0D0) * a_eps * (t1 - rm3 * (t2 + t3 + t4))
            
            vir_lr = 0.0D0

         END SUBROUTINE

         SUBROUTINE aziz_getall(rc, natoms, atoms, cell_h, cell_ih, index_list, n_list, pot, forces, virial)
            ! Calculates the Aziz potential energy and virial and the forces 
            ! acting on all the atoms.
            !
            ! Args:
            !    rc: The cut-off radius.
            !    natoms: The number of atoms in the system.
            !    atoms: A vector holding all the atom positions.
            !    cell_h: The simulation box cell vector matrix.
            !    cell_ih: The inverse of the simulation box cell vector matrix.
            !    index_list: A array giving the last index of n_list that 
            !       gives the neighbours of a given atom.
            !    n_list: An array giving the indices of the atoms that neighbour
            !       the atom determined by index_list.
            !    pot: The total potential energy of the system.
            !    forces: An array giving the forces acting on all the atoms.
            !    virial: The virial tensor, not divided by the volume.

            DOUBLE PRECISION, INTENT(IN) :: rc
            INTEGER, INTENT(IN) :: natoms
            DOUBLE PRECISION, DIMENSION(natoms,3), INTENT(IN) :: atoms
            DOUBLE PRECISION, DIMENSION(3,3), INTENT(IN) :: cell_h
            DOUBLE PRECISION, DIMENSION(3,3), INTENT(IN) :: cell_ih
            INTEGER, DIMENSION(natoms), INTENT(IN) :: index_list
            INTEGER, DIMENSION(natoms*(natoms-1)/2), INTENT(IN) :: n_list
            DOUBLE PRECISION, INTENT(OUT) :: pot
            DOUBLE PRECISION, DIMENSION(natoms,3), INTENT(OUT) :: forces
            DOUBLE PRECISION, DIMENSION(3,3), INTENT(OUT) :: virial

            INTEGER i, j, k, l, start
            DOUBLE PRECISION, DIMENSION(3) :: fij, rij
            DOUBLE PRECISION r2, pot_ij, pot_lr, vir_lr, volume

            forces = 0.0d0
            pot = 0.0d0
            virial = 0.0d0

            start = 1

            DO i = 1, natoms - 1
               ! Only loops over the neigbour list, not all the atoms.
               DO j = start, index_list(i)
                  CALL vector_separation(cell_h, cell_ih, atoms(i,:), atoms(n_list(j),:), rij, r2)
                  IF (r2 < rc*rc) THEN ! Only calculates contributions between neighbouring particles.
                     CALL aziz_fij(rij, sqrt(r2), pot_ij, fij)

                     forces(i,:) = forces(i,:) + fij
                     forces(n_list(j),:) = forces(n_list(j),:) - fij
                     pot = pot + pot_ij
                     DO k = 1, 3
                        DO l = k, 3
                           ! Only the upper triangular elements calculated.
                           virial(k,l) = virial(k,l) + fij(k)*rij(l)
                        ENDDO
                     ENDDO
                  ENDIF
               ENDDO
               start = index_list(i) + 1
            ENDDO

            ! Assuming an upper-triangular vector matrix for the simulation box.
            volume = cell_h(1,1)*cell_h(2,2)*cell_h(3,3)
            CALL aziz_longrange(rc, natoms, volume, pot_lr, vir_lr)
            pot = pot + pot_lr
            DO k = 1, 3
               virial(k,k) = virial(k,k) + vir_lr
            ENDDO

         END SUBROUTINE

      END MODULE
