/*
    ================================================================
    3D STENCIL WITH MPI + HALO EXCHANGE (Assignment 2)
    ================================================================

    What this program is doing:

    We simulate a big 3D grid, but instead of one process handling everything,
    we split the grid across multiple MPI processes. Each process works on
    its own small chunk, and at every step they exchange boundary data with
    neighbours to keep everything consistent.

    Over multiple time steps, we:
    - update values using a stencil computation
    - track how many values cross a given isovalue threshold

    Core ideas behind the implementation:

    1) Domain decomposition 
       The full 3D grid is divided into a px × py × pz grid of processes.
       Each process owns a block of size nx × ny × nz.

       So overall:
       global size = (px*nx) × (py*ny) × (pz*nz)

    2) Halo / ghost cells
       Each local block is padded with h extra layers on all sides.

    3) Multiple fields
       We don’t just simulate one grid — we simulate F independent grids

    4) Time stepping
       The whole stencil update runs for T iterations.
       Each iteration depends on the previous one.

    5) Isovalue counting
       After updates, we count how many points cross a given threshold
       (the isovalue).

       Each process computes its own count, and then we combine everything
       using MPI_Reduce to get the global result.

    
    Important performance tricks used here:

    1) Non-blocking communication (MPI_Isend / MPI_Irecv)
       We don’t wait for communication to finish immediately.

       Instead:
       - start halo exchange
       - compute interior points (which don’t need halo data)
       - then wait and compute boundary points


    2) Double buffering (cur and nxt arrays):
       - read from cur
       - write results into nxt
       - swap them after each step


    3) MPI derived datatypes (zero-copy halos)
       Instead of manually copying boundary data into buffers,
       we describe their layout using MPI_Type_indexed.

       MPI then directly sends the correct strided memory regions.

    ================================================================
*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define IDX(x, y, z, NX, NY) ((x) + (NX) * ((y) + (NY) * (z)))

int main(int argc, char *argv[])
{


    MPI_Init(&argc, &argv);

    int myrank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    /*
        If arguments are missing, only rank 0 prints usage and we exit.
    */
    if (argc < 13)
    {
        if (myrank == 0)
            fprintf(stderr, "Usage: %s d ppn px py pz nx ny nz T seed F isovalue\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    /* Read all parameters */
    int d = atoi(argv[1]);
    int ppn = atoi(argv[2]);
    int px = atoi(argv[3]);
    int py = atoi(argv[4]);
    int pz = atoi(argv[5]);
    int nx = atoi(argv[6]);
    int ny = atoi(argv[7]);
    int nz = atoi(argv[8]);
    int T = atoi(argv[9]);
    int seed = atoi(argv[10]);
    int F = atoi(argv[11]);
    double isovalue = atof(argv[12]);

    /*
        Halo width calculation
        For a stencil of size d, we need h layers of neighbours.
    */
    int h = (d - 1) / 6;

    /*
        Process grid coordinates (rx, ry, rz)
        We map linear rank to 3D grid coordinates.
    */
    int rx = myrank % px;
    int ry = (myrank / px) % py;
    int rz = myrank / (px * py);

    /*
        Find neighbours in 6 directions
        If we are at a boundary, neighbour = MPI_PROC_NULL.
        MPI automatically ignores sends/receives to PROC_NULL
    */
    int nbr_xm = (rx > 0) ? myrank - 1 : MPI_PROC_NULL;
    int nbr_xp = (rx < px - 1) ? myrank + 1 : MPI_PROC_NULL;
    int nbr_ym = (ry > 0) ? myrank - px : MPI_PROC_NULL;
    int nbr_yp = (ry < py - 1) ? myrank + px : MPI_PROC_NULL;
    int nbr_zm = (rz > 0) ? myrank - px * py : MPI_PROC_NULL;
    int nbr_zp = (rz < pz - 1) ? myrank + px * py : MPI_PROC_NULL;

    /*
        Extended grid size
        We pad the local grid with halo layers on all sides.
        ex, ey, ez = full size including ghost cells
    */
    int ex = nx + 2 * h;
    int ey = ny + 2 * h;
    int ez = nz + 2 * h;

    /* Number of actual interior points vs total padded size */
    long arrSize = (long)nx * ny * nz;
    long extSize = (long)ex * ey * ez;

    /*
        Allocate memory for fields

        We keep F independent 3D grids.
        cur is the current timestep
        nxt is the next timestep 
        We use calloc so everything starts at 0.
    */
    double **cur = (double **)malloc(F * sizeof(double *));
    double **nxt = (double **)malloc(F * sizeof(double *));
    for (int i = 0; i < F; i++)
    {
        cur[i] = (double *)calloc(extSize, sizeof(double));
        nxt[i] = (double *)calloc(extSize, sizeof(double));
    }

    /*
        Initialize interior values randomly
    */
    srand(seed);
    for (int fi = 0; fi < F; fi++)
    {
        for (long j = 0; j < arrSize; j++)
        {

            /* Convert flat index to 3D local coordinates */
            int lx = (int)(j % nx);
            int ly = (int)((j / nx) % ny);
            int lz = (int)(j / ((long)nx * ny));

            /* Shift by +h to place inside padded grid */
            long idx = IDX(lx + h, ly + h, lz + h, ex, ey);

            /* Assign some pseudo-random value */
            cur[fi][idx] = (double)rand() * (myrank + 1) / (110426.0 + fi + j);
        }
    }

    /*
        MPI Derived Datatypes: 
        Instead of manually packing halo regions into buffers,
        we describe their layout directly using MPI_Type_indexed.
        This lets MPI directly read/write strided memory regions.
    */

    /* X-face: slices along x-direction (h thickness) */
    MPI_Datatype type_x_face, type_y_face, type_z_face;
    int idx, z, y;

    int num_blocks_x = ny * nz;
    int *blens_x = (int *)malloc(num_blocks_x * sizeof(int));
    int *disps_x = (int *)malloc(num_blocks_x * sizeof(int));

    idx = 0;
    for (z = 0; z < nz; z++)
    {
        for (y = 0; y < ny; y++)
        {

            /* each block is h consecutive elements in x direction */
            blens_x[idx] = h;

            /* displacement jumps across rows and layers */
            disps_x[idx] = y * ex + z * ex * ey;
            idx++;
        }
    }

    /* commit datatype so MPI can use it */
    MPI_Type_indexed(num_blocks_x, blens_x, disps_x, MPI_DOUBLE, &type_x_face);
    MPI_Type_commit(&type_x_face);

    free(blens_x);
    free(disps_x);

    /*
        Y-face: similar idea but now thickness is along y
    */
    int num_blocks_y = h * nz;
    int *blens_y = (int *)malloc(num_blocks_y * sizeof(int));
    int *disps_y = (int *)malloc(num_blocks_y * sizeof(int));

    idx = 0;
    for (z = 0; z < nz; z++)
    {
        for (y = 0; y < h; y++)
        {

            blens_y[idx] = nx;
            disps_y[idx] = y * ex + z * ex * ey;
            idx++;
        }
    }

    MPI_Type_indexed(num_blocks_y, blens_y, disps_y, MPI_DOUBLE, &type_y_face);
    MPI_Type_commit(&type_y_face);

    free(blens_y);
    free(disps_y);

    /*
        Z-face: slices along z direction
    */
    int num_blocks_z = ny * h;
    int *blens_z = (int *)malloc(num_blocks_z * sizeof(int));
    int *disps_z = (int *)malloc(num_blocks_z * sizeof(int));

    idx = 0;
    for (z = 0; z < h; z++)
    {
        for (y = 0; y < ny; y++)
        {

            blens_z[idx] = nx;
            disps_z[idx] = y * ex + z * ex * ey;
            idx++;
        }
    }

    MPI_Type_indexed(num_blocks_z, blens_z, disps_z, MPI_DOUBLE, &type_z_face);
    MPI_Type_commit(&type_z_face);

    free(blens_z);
    free(disps_z);

    /*
        Counting arrays
        local_count  -> per-process counts
        global_count -> result after MPI_Reduce
    */
    long *local_count = (long *)calloc(F, sizeof(long));
    long *global_count = (long *)calloc(F, sizeof(long));

    /*
        Message tags for each direction
    */
    int TAG_XM = 10, TAG_XP = 11, TAG_YM = 12, TAG_YP = 13, TAG_ZM = 14, TAG_ZP = 15;

    /*
        Requests and status arrays for non-blocking communication
        Max = 12 per field (6 recv + 6 send)
    */
    MPI_Request *reqs = (MPI_Request *)malloc(12 * F * sizeof(MPI_Request));
    MPI_Status *stats = (MPI_Status *)malloc(12 * F * sizeof(MPI_Status));

    /*
        Synchronize before timing
        Ensures all processes start measuring at the same time
    */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();
/*
    MAIN TIME LOOP
    We run T time steps. In each step we need to:
    1. Get boundary data from neighbours (halo exchange)
    2. Compute the stencil update for all points
    3. Count isovalue crossings and reduce to root
    
    Key optimization: we start the halo exchange first (non-blocking), then compute interior points (which don't need ghost data) while
    communication happens in background, then wait for halos to arrive and finally compute the boundary points */
    for (int t = 0; t < T; t++) {

        int req_count = 0;

        /* 
        Halo Exchange (Non-blocking)
    
        Before computing the stencil, each process needs h layers of data from its 6 face neighbours (±x, ±y, ±z). 
        We post all Irecvs first, then Isends. Since calls are non-blocking, we immediately return and do useful work.
        
        The padded array has ghost shells of width h around the interior. Receives land directly into those ghost shells. 
        Sends go from the inner boundary slice (first/last h layers of actual interior data).

        Tags are directional so messages don't get mixed up:
        a message sent toward -x is tagged TAG_XM, and the receiver on that side expects TAG_XM. MPI_PROC_NULL handles the case
        where there is no neighbour (global domain boundary) — those calls are silently ignored by MPI. */
        for (int fi = 0; fi < F; fi++) {
            double *A = cur[fi];

            /* Receive ghost data into the outer halo shells */
            MPI_Irecv(&A[IDX(0, h, h, ex, ey)], 1, type_x_face, nbr_xm, TAG_XP, MPI_COMM_WORLD, &reqs[req_count++]);
            MPI_Irecv(&A[IDX(nx+h, h, h, ex, ey)], 1, type_x_face, nbr_xp, TAG_XM, MPI_COMM_WORLD, &reqs[req_count++]);
            
            MPI_Irecv(&A[IDX(h, 0, h, ex, ey)], 1, type_y_face, nbr_ym, TAG_YP, MPI_COMM_WORLD, &reqs[req_count++]);
            MPI_Irecv(&A[IDX(h, ny+h, h, ex, ey)], 1, type_y_face, nbr_yp, TAG_YM, MPI_COMM_WORLD, &reqs[req_count++]);
            
            MPI_Irecv(&A[IDX(h, h, 0, ex, ey)], 1, type_z_face, nbr_zm, TAG_ZP, MPI_COMM_WORLD, &reqs[req_count++]);
            MPI_Irecv(&A[IDX(h, h, nz+h, ex, ey)], 1, type_z_face, nbr_zp, TAG_ZM, MPI_COMM_WORLD, &reqs[req_count++]);

            /* Send our boundary slices to neighbours */
            MPI_Isend(&A[IDX(h, h, h, ex, ey)], 1, type_x_face, nbr_xm, TAG_XM, MPI_COMM_WORLD, &reqs[req_count++]);
            MPI_Isend(&A[IDX(nx, h, h, ex, ey)], 1, type_x_face, nbr_xp, TAG_XP, MPI_COMM_WORLD, &reqs[req_count++]);
            
            MPI_Isend(&A[IDX(h, h, h, ex, ey)], 1, type_y_face, nbr_ym, TAG_YM, MPI_COMM_WORLD, &reqs[req_count++]);
            MPI_Isend(&A[IDX(h, ny, h, ex, ey)], 1, type_y_face, nbr_yp, TAG_YP, MPI_COMM_WORLD, &reqs[req_count++]);
            
            MPI_Isend(&A[IDX(h, h, h, ex, ey)], 1, type_z_face, nbr_zm, TAG_ZM, MPI_COMM_WORLD, &reqs[req_count++]);
            MPI_Isend(&A[IDX(h, h, nz, ex, ey)], 1, type_z_face, nbr_zp, TAG_ZP, MPI_COMM_WORLD, &reqs[req_count++]);
        }

        /* Interior Stencil (overlaps with communication)
        Points at least h away from all local edges are "interior" — their stencil neighbours are all within this process's own data,
        so we don't need ghost data for them. We compute these now while the halo exchange runs in the background.
        For each point we sum contributions from up to h neighbours in each of the 6 directions,
        but only if that neighbour exists in the global domain (boundary processes have fewer neighbours).
        The denominator cnt tracks how many actually contributed, so boundary points are averaged correctly.*/
        for (int fi = 0; fi < F; fi++) {
            double *A = cur[fi];
            double *B = nxt[fi];

            for (int lz = h; lz < nz - h; lz++) {           
                for (int ly = h; ly < ny - h; ly++) {
                    for (int lx = h; lx < nx - h; lx++) {
                        int gx = rx*nx + lx; 
                        int gy = ry*ny + ly;
                        int gz = rz*nz + lz;
                        int gnx = px*nx, gny = py*ny, gnz = pz*nz;

                        double sum = A[IDX(lx+h, ly+h, lz+h, ex, ey)];
                        int cnt = 1;

                        for (int s = 1; s <= h; s++) {
                            if (gx - s >= 0) { sum += A[IDX(lx+h-s, ly+h, lz+h, ex, ey)]; cnt++; }
                            if (gx + s < gnx){ sum += A[IDX(lx+h+s, ly+h, lz+h, ex, ey)]; cnt++; }
                        }
                        for (int s = 1; s <= h; s++) {
                            if (gy - s >= 0) { sum += A[IDX(lx+h, ly+h-s, lz+h, ex, ey)]; cnt++; }
                            if (gy + s < gny){ sum += A[IDX(lx+h, ly+h+s, lz+h, ex, ey)]; cnt++; }
                        }
                        for (int s = 1; s <= h; s++) {
                            if (gz - s >= 0) { sum += A[IDX(lx+h, ly+h, lz+h-s, ex, ey)]; cnt++; }
                            if (gz + s < gnz){ sum += A[IDX(lx+h, ly+h, lz+h+s, ex, ey)]; cnt++; }
                        }

                        B[IDX(lx+h, ly+h, lz+h, ex, ey)] = sum / cnt;
                    }
                }
            }
        }

        /* Wait for halo exchange to complete before touching boundary points */
        MPI_Waitall(req_count, reqs, stats);
/*
--------------------------------------------------------------
         *  Compute BOUNDARY points (those needing halo data)

         Boundary smoothing:
 * -------------------
 * For each field, we update only the boundary points of the 3D grid.
 *
 * Logic:
 * - Loop over all grid points in the local subdomain.
 * - Skip the inner points (at least h cells away from the boundary).
 * - For each boundary point:
 *   - Add values of neighbors in ±x, ±y, ±z directions,
 *     up to distance h, but only if those neighbors lie inside
 *     the global grid.
 *   - Compute the average of the point and its valid neighbors.
 * - Store the averaged result in the next array (B).
 * - Each process computes its local boundary smoothing independently.
 
         * ------------------------------------------------------------ */
        for (int fi = 0; fi < F; fi++) {
            double *A = cur[fi];
            double *B = nxt[fi];

            for (int lz = 0; lz < nz; lz++) {
                for (int ly = 0; ly < ny; ly++) {
                    for (int lx = 0; lx < nx; lx++) {
                           // Skiping the inner points.
                           
                        if (lx >= h && lx < nx-h &&
                            ly >= h && ly < ny-h &&
                            lz >= h && lz < nz-h)
                            continue; 

                        int gx = rx*nx + lx;
                        int gy = ry*ny + ly;
                        int gz = rz*nz + lz;
                        int gnx = px*nx, gny = py*ny, gnz = pz*nz;

                        double sum = A[IDX(lx+h, ly+h, lz+h, ex, ey)];
                        int cnt = 1;

                        for (int s = 1; s <= h; s++) {
                            if (gx - s >= 0) { sum += A[IDX(lx+h-s, ly+h, lz+h, ex, ey)]; cnt++; }
                            if (gx + s < gnx){ sum += A[IDX(lx+h+s, ly+h, lz+h, ex, ey)]; cnt++; }
                        }
                        for (int s = 1; s <= h; s++) {
                            if (gy - s >= 0) { sum += A[IDX(lx+h, ly+h-s, lz+h, ex, ey)]; cnt++; }
                            if (gy + s < gny){ sum += A[IDX(lx+h, ly+h+s, lz+h, ex, ey)]; cnt++; }
                        }
                        for (int s = 1; s <= h; s++) {
                            if (gz - s >= 0) { sum += A[IDX(lx+h, ly+h, lz+h-s, ex, ey)]; cnt++; }
                            if (gz + s < gnz){ sum += A[IDX(lx+h, ly+h, lz+h+s, ex, ey)]; cnt++; }
                        }

                        B[IDX(lx+h, ly+h, lz+h, ex, ey)] = sum / cnt;
                    }
                }
            }
        }

        /* 
         Swap cur and nxt pointers for the next iteration ,as it costs nothing and aviods copying arrys.
         */
        double **tmp = cur;
        cur = nxt;
        nxt = tmp;

       
/* 
        Count isovalue crossings:
        For each field, we count how many times the scalar value crosses
        a given isovalue along the x, y, and z directions.
        
        Idea:
        At each grid point, compare its value with its neighbor
        in +x, +y, +z directions.
        If the values lie on opposite sides of the isovalue,
        then a crossing occurs.
        Each process computes its local count for its subdomain.         
        */
        for (int fi = 0; fi < F; fi++) local_count[fi] = 0;

        for (int fi = 0; fi < F; fi++) {
            double *A = cur[fi];
            long cnt = 0;

            for (int lz = 0; lz < nz; lz++) {
                for (int ly = 0; ly < ny; ly++) {
                    for (int lx = 0; lx < nx; lx++) {
                        int gx = rx*nx + lx;
                        int gy = ry*ny + ly;
                        int gz = rz*nz + lz;
                        int gnx = px*nx, gny = py*ny, gnz = pz*nz;

                        double v0 = A[IDX(lx+h, ly+h, lz+h, ex, ey)];
                        double d0 = v0 - isovalue;

                        if (gx + 1 < gnx) {
                            double v1 = A[IDX(lx+h+1, ly+h, lz+h, ex, ey)];
                            if (d0 * (v1 - isovalue) < 0.0) cnt++;
                        }
                        if (gy + 1 < gny) {
                            double v1 = A[IDX(lx+h, ly+h+1, lz+h, ex, ey)];
                            if (d0 * (v1 - isovalue) < 0.0) cnt++;
                        }
                        if (gz + 1 < gnz) {
                            double v1 = A[IDX(lx+h, ly+h, lz+h+1, ex, ey)];
                            if (d0 * (v1 - isovalue) < 0.0) cnt++;
                        }
                    }
                }
            }
            local_count[fi] = cnt;
        }

        /* 
        Reduce isovalue counts to root:
        Each process has a local count of crossings.
        We sum these across all processes using MPI_Reduce.
        
        Result:
        global_count contains total crossings for each field
        only root process (rank 0) receives final result
        */
        MPI_Reduce(local_count, global_count, F, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        if (myrank == 0) {
            for (int fi = 0; fi < F; fi++) {
                printf("%ld", global_count[fi]);
                if (fi < F-1) printf(" ");
            }
            printf("\n");
        }

    } /* end time loop */

    /* 
    Timing and synchronization:
    Ensure all processes finish computation before measuring time.
    Only root prints total execution time.
    */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();
    if (myrank == 0) {
        printf("%f\n", t_end - t_start);
    }

   /* 
   Cleanup Types and Memory: 
   Free all allocated MPI datatypes and dynamically allocated memory
   to avoid memory leaks.
   */ 
    MPI_Type_free(&type_x_face);
    MPI_Type_free(&type_y_face);
    MPI_Type_free(&type_z_face);

    for (int i = 0; i < F; i++) { free(cur[i]); free(nxt[i]); }
    free(cur); free(nxt);
    free(reqs); free(stats);
    free(local_count); free(global_count);

    MPI_Finalize();
    return 0;
}