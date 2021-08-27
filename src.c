/*
Source Code of HaloExchange
Authors : Shobhit Sinha
Last Updated on : 20:06, 23rd Feb, 2021
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "mpi.h"

#define CONST_NDIMS 2
#define CONST_NO_REORDER 0

/*Constants to denote the buffer locations, used in comm_by_pack()
*/
#define CONST_TOP_ROW 0
#define CONST_BOT_ROW 1
#define CONST_LEFT_COL 2
#define CONST_RIGHT_COL 3

/*These Constants represent the location of a process in the Square Topology.
  They will be returned by the process_location() and will be used to determine which buffers need to be used for computation.
*/
#define INTERIOR_PROCESS 10
#define TOP_LEFT_CORNER 11
#define TOP_RIGHT_CORNER 12  
#define BOT_LEFT_CORNER 13
#define BOT_RIGHT_CORNER 14
#define TOP_ROW_NOT_CORNER 15
#define BOT_ROW_NOT_CORNER 16
#define LEFT_COL_NOT_CORNER 17
#define RIGHT_COL_NOT_CORNER 18


/*These are Auxiliiary functions*/
int process_location(int *coordinates, int dim_limit);
void initialize_randomly(int N, double **matrix, int rank);
double** alloc_2D_array_memory(int R, int C);

/*
PROTOTYPES FOR THE COMMUNICATION FUNCTIONS
*/
void comm_by_single_element(int N, double **matrix, double *top, double *bottom, double *left, double *right, MPI_Comm comm2D);
void comm_by_pack(int N, double **matrix, double** recv, double *top, double *bottom, double *left, double *right, MPI_Comm comm2D);
void comm_by_derived(int N, double **matrix, double *top, double *bottom, double *left, double *right, MPI_Comm comm2D, MPI_Datatype rowtype, MPI_Datatype coltype);

/*PROTOTYPE FOR THE COMPUTATION FUNCTION
*/
void computation(int N, double **matrix, double **mat_copy, double *top, double *bottom, double *left, double *right, int* coordinates);

int main(int argc, char* argv[])
{
	int N = atoi(argv[1]); /*gets the  value of N from Command Line*/
	int num_time_steps = atoi(argv[2]); /* gets the value of num_time_steps from Command Line*/
	double start_time, end_time, total_time, max_time; /*Timing Variables*/
	double max_single_time = 0, max_pack_time = 0, max_vector_time = 0;

	double **matrix, **mat_copy; 
	double *top_row_buffer, *bottom_row_buffer, *left_column_buffer, *right_column_buffer;
	MPI_Init(&argc, &argv);

	/* Getting CONTIGUOUS Memory for the matrix and the copy matrix*/
	matrix = (double**)malloc(N*sizeof(double*));
	matrix[0] = (double*)malloc(N*N*sizeof(double));
	for(int i = 1; i< N; i++)
		matrix[i] = &(matrix[0][i*N]);
	/*To Free above use free(matrix[0]);free(matrix);*/
	mat_copy = (double**)malloc(N*sizeof(double*));
	mat_copy[0] = (double*)malloc(N*N*sizeof(double));
	for(int i = 1; i< N; i++)
		mat_copy[i] = &(mat_copy[0][i*N]);

	/*Getting memory for the buffers which will store incoming values from the neighbours*/
	top_row_buffer = (double*)malloc(N*sizeof(double));
	bottom_row_buffer = (double*)malloc(N*sizeof(double));
	left_column_buffer = (double*)malloc(N*sizeof(double));
	right_column_buffer = (double*)malloc(N*sizeof(double));

	/*This extra memory will be required when we use MPI_pack*/
	double **recv_buff = alloc_2D_array_memory(4,N);

	int global_rank, global_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &global_size);


	/*
	 Creating Two Derived Vector Types to store rows and columns and committing them
	 */
	MPI_Datatype rowtype;
	MPI_Datatype coltype;
    MPI_Type_vector(N, 1, 1, MPI_DOUBLE, &rowtype);
	MPI_Type_vector(N, 1, N, MPI_DOUBLE, &coltype);

	MPI_Type_commit(&rowtype);
	MPI_Type_commit(&coltype);

	// Initializing parameters for Cartesian Topology
	MPI_Comm comm2D;
	const int dim[] = {(int)(sqrt(global_size)),(int)(sqrt(global_size))};
	const int wrap_around[] = {0,0};
	MPI_Cart_create(MPI_COMM_WORLD, CONST_NDIMS, dim, wrap_around, CONST_NO_REORDER, &comm2D );

	// Getting current coordinates of the process
	int coordinates[CONST_NDIMS];
	MPI_Cart_coords(comm2D, global_rank, CONST_NDIMS, coordinates);

	/****************************************
	****************************************
	CODE WHEN COMMUNICATION IS THROUGH SENDRECV ONLY 1 element at a time
	****************************************
	*****************************************/
	//num_time_steps = 1; FOR DEBUGGING, when you use only 1 time step to test.
	total_time = 0;
	initialize_randomly(N, matrix, global_rank);
	for(int ct_count = 0; ct_count < num_time_steps; ct_count++)
	{
		start_time = MPI_Wtime();
		comm_by_single_element(N, matrix, top_row_buffer, bottom_row_buffer, left_column_buffer, right_column_buffer, comm2D);
	
		/*Note that MPI_barrier is not required here due to the implicit blocking nature of MPI_Sendrecv.
		If a process has received the data from neighbours and communicated data to the neighbours, it is free to proceed towards computation
		*/
		computation(N, matrix, mat_copy, top_row_buffer, bottom_row_buffer, left_column_buffer, right_column_buffer, coordinates);	
		end_time = MPI_Wtime();
		total_time  = total_time + (end_time - start_time);

	}
	MPI_Reduce (&total_time, &max_single_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	
	/****************************************
	****************************************
	CODE WHEN COMMUNICATION IS THROUGH SENDRECV and PACK/UNPACK
	****************************************
	*****************************************/
	initialize_randomly(N, matrix, global_rank);
	total_time = 0;
	for(int ct_count = 0; ct_count < num_time_steps; ct_count++)
	{
		start_time = MPI_Wtime();
		comm_by_pack(N, matrix, recv_buff, top_row_buffer, bottom_row_buffer, left_column_buffer, right_column_buffer, comm2D);
		/*Note that MPI_barrier is not required here due to the implicit blocking nature of MPI_Sendrecv.
		If a process has received the data from neighbours and communicated data to the neighbours, it is free to proceed towards computation
		*/
		computation(N, matrix, mat_copy, top_row_buffer, bottom_row_buffer, left_column_buffer, right_column_buffer, coordinates);	
		end_time = MPI_Wtime();
		total_time  = total_time + (end_time - start_time);

	}
	MPI_Reduce (&total_time, &max_pack_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	
	/****************************************
	*****************************************
	CODE WHEN COMMUNICATION IS THROUGH SENDRECV and DERIVED DATATYPE VECTOR
	*****************************************
	****************************************/
	initialize_randomly(N, matrix, global_rank);
	total_time = 0;

	for(int ct_count = 0; ct_count < num_time_steps; ct_count++)
	{
		start_time = MPI_Wtime();
		comm_by_derived(N, matrix, top_row_buffer, bottom_row_buffer, left_column_buffer, right_column_buffer, comm2D, rowtype, coltype);
		/*Note that MPI_barrier is not required here due to the implicit blocking nature of MPI_Sendrecv.
		If a process has received the data from neighbours and communicated data to the neighbours, it is free to proceed towards computation
		*/

		computation(N, matrix, mat_copy, top_row_buffer, bottom_row_buffer, left_column_buffer, right_column_buffer, coordinates);
		end_time = MPI_Wtime();
		total_time  = total_time + (end_time - start_time);
	
	}
	MPI_Reduce (&total_time, &max_vector_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	if(global_rank==0)
	{
		printf("%d,%d,a)Single,%.6f\n",global_size,N,max_single_time);
		printf("%d,%d,b)Pack,%.6f\n",global_size,N,max_pack_time);
		printf("%d,%d,c)Vector,%.6f\n",global_size,N,max_vector_time);

	}
	
	/*Freeing memory and Data Types*/
	free(matrix[0]);
	free(matrix);
	free(mat_copy[0]);
	free(mat_copy);
	free(top_row_buffer);
	free(bottom_row_buffer);
	free(left_column_buffer);
	free(right_column_buffer);
	
	MPI_Type_free (&rowtype);
	MPI_Type_free (&coltype);
	MPI_Finalize();
}
/*
MPI_PROC_NULL which is basically -1, is what the MPI_Cart_Shift() will return if there is no source and destination. Which takes care of the boundary conditions automatically 
*/
void comm_by_single_element(int N, double **matrix, double *top, double *bottom, double *left, double *right, MPI_Comm comm2D){
	int source, dest;
	/* 
	LEFT TO RIGHT COMM	
	*/
	MPI_Cart_shift (comm2D, 1, 1, &source, &dest);
	for(int i = 0; i< N; i++){
	 	MPI_Sendrecv(&matrix[i][N-1], 1, MPI_DOUBLE, dest, 0 , &left[i], 1, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	/*
	RIGHT TO LEFT COMM
	*/
	MPI_Cart_shift (comm2D, 1, -1, &source, &dest);
	for(int i = 0; i< N; i++){
	 	MPI_Sendrecv(&matrix[i][0], 1, MPI_DOUBLE, dest, 0 , &right[i], 1, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}	
	/*
	TOP TO BOTTOM COMM
	*/
	MPI_Cart_shift (comm2D, 0, 1, &source, &dest);
	for(int i = 0; i< N; i++){
	 	MPI_Sendrecv(&matrix[N-1][i], 1, MPI_DOUBLE, dest, 0 , &top[i], 1, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	/*
	BOTTOM TO TOP COMM
	*/
	MPI_Cart_shift (comm2D, 0, -1, &source, &dest);
	for(int i = 0; i< N; i++){
	 	MPI_Sendrecv(&matrix[0][i], 1, MPI_DOUBLE, dest, 0 , &bottom[i], 1, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}	

}
void comm_by_pack(int N, double **matrix, double **recv_buff, double *top, double *bottom, double *left, double *right, MPI_Comm comm2D){
	int source, dest;
	int send_position[] = {0,0,0,0};
	int recv_position[] = {0,0,0,0};
	/*
	PACKING THE FOUR BOUNDARY REGIONS
	ThE top and bottom rows can be packed directly.
	The Left and right column need to be packed using a LOOP 
	*/
	MPI_Pack(matrix[0], N, MPI_DOUBLE, top, N*sizeof(double), &send_position[CONST_TOP_ROW], MPI_COMM_WORLD);
	MPI_Pack(matrix[N-1], N, MPI_DOUBLE, bottom, N*sizeof(double), &send_position[CONST_BOT_ROW], MPI_COMM_WORLD);

	for(int i = 0; i< N; i++){
		MPI_Pack(&matrix[i][0], 1, MPI_DOUBLE, left, N*sizeof(double), &send_position[CONST_LEFT_COL], MPI_COMM_WORLD);
		MPI_Pack(&matrix[i][N-1], 1, MPI_DOUBLE, right, N*sizeof(double), &send_position[CONST_RIGHT_COL], MPI_COMM_WORLD);
	}
	/* 
	LEFT TO RIGHT COMM	
	*/
	MPI_Cart_shift (comm2D, 1, 1, &source, &dest);
	MPI_Sendrecv(right, send_position[CONST_RIGHT_COL], MPI_PACKED, dest, 0 , recv_buff[CONST_LEFT_COL], send_position[CONST_LEFT_COL], MPI_PACKED, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	/*
	RIGHT TO LEFT COMM
	*/
	MPI_Cart_shift (comm2D, 1, -1, &source, &dest);
	MPI_Sendrecv(left, send_position[CONST_LEFT_COL], MPI_PACKED, dest, 0 , recv_buff[CONST_RIGHT_COL], send_position[CONST_RIGHT_COL], MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	/*
	TOP TO BOTTOM COMM
	*/
	MPI_Cart_shift (comm2D, 0, 1, &source, &dest);
	MPI_Sendrecv(bottom, send_position[CONST_BOT_ROW], MPI_PACKED, dest, 0 , recv_buff[CONST_TOP_ROW],send_position[CONST_TOP_ROW], MPI_PACKED, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	/*
	BOTTOM TO TOP COMM
	*/
	MPI_Cart_shift (comm2D, 0, -1, &source, &dest);
	MPI_Sendrecv(top, send_position[CONST_TOP_ROW], MPI_PACKED, dest, 0 , recv_buff[CONST_BOT_ROW],send_position[CONST_BOT_ROW], MPI_PACKED, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	/*
	 UNPACKING THE BOUNDARY REGION
	*/
	MPI_Unpack(recv_buff[CONST_TOP_ROW],send_position[CONST_TOP_ROW],&recv_position[CONST_TOP_ROW], top, N, MPI_DOUBLE, MPI_COMM_WORLD);
	MPI_Unpack(recv_buff[CONST_BOT_ROW],send_position[CONST_BOT_ROW],&recv_position[CONST_BOT_ROW], bottom, N, MPI_DOUBLE, MPI_COMM_WORLD);
	MPI_Unpack(recv_buff[CONST_LEFT_COL],send_position[CONST_LEFT_COL],&recv_position[CONST_LEFT_COL], left, N, MPI_DOUBLE, MPI_COMM_WORLD);
	MPI_Unpack(recv_buff[CONST_RIGHT_COL],send_position[CONST_RIGHT_COL],&recv_position[CONST_RIGHT_COL], right, N, MPI_DOUBLE, MPI_COMM_WORLD);
}
void comm_by_derived(int N, double **matrix, double *top, double *bottom, double *left, double *right, MPI_Comm comm2D, MPI_Datatype rowtype, MPI_Datatype coltype){
	int source, dest;
	/* 
	LEFT TO RIGHT COMM	
	*/
	MPI_Cart_shift (comm2D, 1, 1, &source, &dest);
	MPI_Sendrecv(&matrix[0][N-1], 1, coltype, dest, 0 , left, N, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	/*
	RIGHT TO LEFT COMM
	*/
	MPI_Cart_shift (comm2D, 1, -1, &source, &dest);
	MPI_Sendrecv(&matrix[0][0], 1, coltype, dest, 0 , right, N, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	/*
	TOP TO BOTTOM COMM
	*/
	MPI_Cart_shift (comm2D, 0, 1, &source, &dest);
	MPI_Sendrecv(&matrix[N-1][0], 1, rowtype, dest, 0 , top, N, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	/*
	BOTTOM TO TOP COMM
	*/
	MPI_Cart_shift (comm2D, 0, -1, &source, &dest);
	MPI_Sendrecv(&matrix[0][0], 1, rowtype, dest, 0 , bottom, N, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}
void computation(int N, double **matrix, double **mat_copy, double *top, double *bottom, double *left, double *right, int* coordinates){

	/*Interior N-1 X N-1 can be updated normally*/
	for(int i = 1; i<N-1; i++)
		for (int j = 1; j < N-1; ++j)
			mat_copy[i][j] = (matrix[i-1][j] + matrix[i+1][j]+ matrix[i][j-1]+ matrix[i][j+1])/4;
	
	int dim_limit = (int)(sqrt(N));
	int location_status = process_location(coordinates, dim_limit-1);
	if(location_status==INTERIOR_PROCESS)
	{
		/*All Buffers need to be used*/

		/*Updating Corner Points*/
		mat_copy[0][0] = (top[0] + left[0] + matrix[0][1] + matrix[1][0])/4;
		mat_copy[0][N-1] = (top[N-1] + right[0] + matrix[1][N-1] + matrix[0][N-2])/4;
		mat_copy[N-1][0] = (bottom[0] + left[N-1] + matrix[N-2][0] + matrix[N-1][1])/4;
		mat_copy[N-1][N-1] = (bottom[N-1] + right[N-1] + matrix[N-1][N-2] + matrix[N-2][N-1])/4;
		
		/*Updating borders*/
		for(int i = 1; i<N-1; i++)
			mat_copy[0][i] = (top[i] + matrix[1][i]+ matrix[0][i-1]+ matrix[0][i+1])/4;
		for(int i = 1; i<N-1; i++)
			mat_copy[N-1][i] = (bottom[i] + matrix[N-2][i]+ matrix[N-1][i-1]+ matrix[N-1][i+1])/4;
		for(int i = 1; i<N-1; i++)
			mat_copy[i][0] = (left[i] + matrix[i][1]+ matrix[i-1][0]+ matrix[i+1][0])/4;
		for(int i = 1; i<N-1; i++)
			mat_copy[i][N-1] = (right[i] + matrix[i][N-2]+ matrix[i+1][N-1]+ matrix[i-1][N-1])/4;
	}
	else if(location_status==TOP_LEFT_CORNER)
	{
		/*top and left will not be used*/

		/*Updating Corner Points*/
		mat_copy[0][0] = (matrix[0][1] + matrix[1][0])/2;
		mat_copy[0][N-1] = (right[0] + matrix[1][N-1] + matrix[0][N-2])/3;
		mat_copy[N-1][0] = (bottom[0] + matrix[N-2][0] + matrix[N-1][1])/3;
		mat_copy[N-1][N-1] = (bottom[N-1] + right[N-1] + matrix[N-1][N-2] + matrix[N-2][N-1])/4;

		/*Updating borders*/
		for(int i = 1; i<N-1; i++)
			mat_copy[0][i] = (matrix[0][i-1] + matrix[0][i+1] + matrix[1][i])/3;
		for(int i = 1; i<N-1; i++)
			mat_copy[N-1][i] = (bottom[i] + matrix[N-1][i-1] + matrix[N-1][i+1] + matrix[N-2][i])/4;
		for(int i = 1; i<N-1; i++)
			mat_copy[i][0] = (matrix[i][1] + matrix[i-1][0] + matrix[i+1][0])/3;
		for(int i = 1; i<N-1; i++)
			mat_copy[i][N-1] = (right[i] + matrix[i-1][N-1]+ matrix[i+1][N-1]+ matrix[i][N-2])/4;
	}
	else if(location_status==TOP_RIGHT_CORNER)
	{
		/*top and right will not be used*/
		/*Updating Corner Points*/
		mat_copy[0][0] = ( left[0] + matrix[0][1] + matrix[1][0])/3;
		mat_copy[0][N-1] = (matrix[1][N-1] + matrix[0][N-2])/2;
		mat_copy[N-1][0] = (bottom[0] + left[N-1] + matrix[N-2][0] + matrix[N-1][1])/4;
		mat_copy[N-1][N-1] = (bottom[N-1] +  matrix[N-1][N-2] + matrix[N-2][N-1])/3;
		
		/*Updating borders*/
		for(int i = 1; i<N-1; i++)
			mat_copy[0][i] = (matrix[1][i]+ matrix[0][i-1]+ matrix[0][i+1])/3;
		for(int i = 1; i<N-1; i++)
			mat_copy[N-1][i] = (bottom[i] + matrix[N-2][i]+ matrix[N-1][i-1]+ matrix[N-1][i+1])/4;
		for(int i = 1; i<N-1; i++)
			mat_copy[i][0] = (left[i] + matrix[i][1]+ matrix[i-1][0]+ matrix[i+1][0])/4;
		for(int i = 1; i<N-1; i++)
			mat_copy[i][N-1] = ( matrix[i][N-2]+ matrix[i+1][N-1]+ matrix[i-1][N-1])/3;
		
	}
	else if(location_status==BOT_LEFT_CORNER)
	{
		/*bottom and left will not be used*/
		/*Updating Corner Points*/
		mat_copy[0][0] = (top[0] + matrix[0][1] + matrix[1][0])/3;
		mat_copy[0][N-1] = (top[N-1] + right[0] + matrix[1][N-1] + matrix[0][N-2])/4;
		mat_copy[N-1][0] = ( matrix[N-2][0] + matrix[N-1][1])/2;
		mat_copy[N-1][N-1] = (right[N-1] + matrix[N-1][N-2] + matrix[N-2][N-1])/3;
		
		/*Updating Borders*/
		for(int i = 1; i<N-1; i++)
			mat_copy[0][i] = (top[i] + matrix[1][i]+ matrix[0][i-1]+ matrix[0][i+1])/4;
		for(int i = 1; i<N-1; i++)
			mat_copy[N-1][i] = (matrix[N-2][i]+ matrix[N-1][i-1]+ matrix[N-1][i+1])/3;
		for(int i = 1; i<N-1; i++)
			mat_copy[i][0] = ( matrix[i][1]+ matrix[i-1][0]+ matrix[i+1][0])/3;
		for(int i = 1; i<N-1; i++)
			mat_copy[i][N-1] = (right[i] + matrix[i][N-2]+ matrix[i+1][N-1]+ matrix[i-1][N-1])/4;
		
	}
	else if(location_status==BOT_RIGHT_CORNER)
	{
		/*bottom and right will not be used*/
		/*Updating Corner Points*/
		mat_copy[0][0] = (top[0] + left[0] + matrix[0][1] + matrix[1][0])/4;
		mat_copy[0][N-1] = (top[N-1] + matrix[1][N-1] + matrix[0][N-2])/3;
		mat_copy[N-1][0] = (left[N-1] + matrix[N-2][0] + matrix[N-1][1])/3;
		mat_copy[N-1][N-1] = (matrix[N-1][N-2] + matrix[N-2][N-1])/2;
		
		/*Updating borders*/
		for(int i = 1; i<N-1; i++)
			mat_copy[0][i] = (top[i] + matrix[1][i]+ matrix[0][i-1]+ matrix[0][i+1])/4;
		for(int i = 1; i<N-1; i++)
			mat_copy[N-1][i] = ( matrix[N-2][i]+ matrix[N-1][i-1]+ matrix[N-1][i+1])/3;
		for(int i = 1; i<N-1; i++)
			mat_copy[i][0] = (left[i] + matrix[i][1]+ matrix[i-1][0]+ matrix[i+1][0])/4;
		for(int i = 1; i<N-1; i++)
			mat_copy[i][N-1] = (matrix[i][N-2]+ matrix[i+1][N-1]+ matrix[i-1][N-1])/3;
		
	}
	else if(location_status==TOP_ROW_NOT_CORNER)
	{
		/*Top row will not be used*/

		/*Updating Corner Points*/
		mat_copy[0][0] = (left[0] + matrix[0][1] + matrix[1][0])/3;
		mat_copy[0][N-1] = (right[0] + matrix[1][N-1] + matrix[0][N-2])/3;
		mat_copy[N-1][0] = (bottom[0] + left[N-1] + matrix[N-2][0] + matrix[N-1][1])/4;
		mat_copy[N-1][N-1] = (bottom[N-1] + right[N-1] + matrix[N-1][N-2] + matrix[N-2][N-1])/4;
		
		/*Updating borders*/
		for(int i = 1; i<N-1; i++)
			mat_copy[0][i] = (matrix[1][i]+ matrix[0][i-1]+ matrix[0][i+1])/3;
		for(int i = 1; i<N-1; i++)
			mat_copy[N-1][i] = (bottom[i] + matrix[N-2][i]+ matrix[N-1][i-1]+ matrix[N-1][i+1])/4;
		for(int i = 1; i<N-1; i++)
			mat_copy[i][0] = (left[i] + matrix[i][1]+ matrix[i-1][0]+ matrix[i+1][0])/4;
		for(int i = 1; i<N-1; i++)
			mat_copy[i][N-1] = (right[i] + matrix[i][N-2]+ matrix[i+1][N-1]+ matrix[i-1][N-1])/4;
		
	}
	else if(location_status==BOT_ROW_NOT_CORNER)
	{
		/*bottom will not be used*/

		/*Updating Corner Points*/
		mat_copy[0][0] = (top[0] + left[0] + matrix[0][1] + matrix[1][0])/4;
		mat_copy[0][N-1] = (top[N-1] + right[0] + matrix[1][N-1] + matrix[0][N-2])/4;
		mat_copy[N-1][0] = ( left[N-1] + matrix[N-2][0] + matrix[N-1][1])/3;
		mat_copy[N-1][N-1] = (right[N-1] + matrix[N-1][N-2] + matrix[N-2][N-1])/3;
		
		/*Updating borders*/
		for(int i = 1; i<N-1; i++)
			mat_copy[0][i] = (top[i] + matrix[1][i]+ matrix[0][i-1]+ matrix[0][i+1])/4;
		for(int i = 1; i<N-1; i++)
			mat_copy[N-1][i] = (matrix[N-2][i]+ matrix[N-1][i-1]+ matrix[N-1][i+1])/3;
		for(int i = 1; i<N-1; i++)
			mat_copy[i][0] = (left[i] + matrix[i][1]+ matrix[i-1][0]+ matrix[i+1][0])/4;
		for(int i = 1; i<N-1; i++)
			mat_copy[i][N-1] = (right[i] + matrix[i][N-2]+ matrix[i+1][N-1]+ matrix[i-1][N-1])/4;
		
	}
	else if(location_status==LEFT_COL_NOT_CORNER)
	{
		/*left will not be used*/

		/*Updating Corner Points*/
		mat_copy[0][0] = (top[0] +  matrix[0][1] + matrix[1][0])/3;
		mat_copy[0][N-1] = (top[N-1] + right[0] + matrix[1][N-1] + matrix[0][N-2])/4;
		mat_copy[N-1][0] = (bottom[0] +  matrix[N-2][0] + matrix[N-1][1])/3;
		mat_copy[N-1][N-1] = (bottom[N-1] + right[N-1] + matrix[N-1][N-2] + matrix[N-2][N-1])/4;
		
		/*Updating borders*/
		for(int i = 1; i<N-1; i++)
			mat_copy[0][i] = (top[i] + matrix[1][i]+ matrix[0][i-1]+ matrix[0][i+1])/4;
		for(int i = 1; i<N-1; i++)
			mat_copy[N-1][i] = (bottom[i] + matrix[N-2][i]+ matrix[N-1][i-1]+ matrix[N-1][i+1])/4;
		for(int i = 1; i<N-1; i++)
			mat_copy[i][0] = (matrix[i][1]+ matrix[i-1][0]+ matrix[i+1][0])/3;
		for(int i = 1; i<N-1; i++)
			mat_copy[i][N-1] = (right[i] + matrix[i][N-2]+ matrix[i+1][N-1]+ matrix[i-1][N-1])/4;
	}
	else if(location_status==RIGHT_COL_NOT_CORNER)
	{
		/*right will not be used*/

		/*Updating Corner Points*/
		mat_copy[0][0] = (top[0] + left[0] + matrix[0][1] + matrix[1][0])/4;
		mat_copy[0][N-1] = (top[N-1] + matrix[1][N-1] + matrix[0][N-2])/3;
		mat_copy[N-1][0] = (bottom[0] + left[N-1] + matrix[N-2][0] + matrix[N-1][1])/4;
		mat_copy[N-1][N-1] = (bottom[N-1] + matrix[N-1][N-2] + matrix[N-2][N-1])/3;
		
		/*Updating borders*/
		for(int i = 1; i<N-1; i++)
			mat_copy[0][i] = (top[i] + matrix[1][i]+ matrix[0][i-1]+ matrix[0][i+1])/4;
		for(int i = 1; i<N-1; i++)
			mat_copy[N-1][i] = (bottom[i] + matrix[N-2][i]+ matrix[N-1][i-1]+ matrix[N-1][i+1])/4;
		for(int i = 1; i<N-1; i++)
			mat_copy[i][0] = (left[i] + matrix[i][1]+ matrix[i-1][0]+ matrix[i+1][0])/4;
		for(int i = 1; i<N-1; i++)
			mat_copy[i][N-1] = (matrix[i][N-2]+ matrix[i+1][N-1]+ matrix[i-1][N-1])/3;
	}
	/* Making mat_copy as the new matrix to be used in next computation */ 
	double **temp;
	temp = mat_copy;
	mat_copy = matrix;
	matrix = temp;

}
int process_location(int *coordinates, int dim_limit)
{
	if(coordinates[0] > 0 && coordinates[0] < dim_limit && coordinates[1] > 0 && coordinates[1] < dim_limit)
		return INTERIOR_PROCESS;
	if(coordinates[0] == 0 && coordinates[1] == 0)
		return TOP_LEFT_CORNER;
	if(coordinates[0] == 0 && coordinates[1] == dim_limit)
		return TOP_RIGHT_CORNER;
	if(coordinates[0] == dim_limit && coordinates[1] == 0)
		return BOT_LEFT_CORNER;
	if(coordinates[0] == dim_limit && coordinates[1] == dim_limit)
		return BOT_RIGHT_CORNER;
	if(coordinates[0] == 0)
		return TOP_ROW_NOT_CORNER;
	if(coordinates[0] == dim_limit)
		return BOT_ROW_NOT_CORNER;
	if(coordinates[1] == 0)
		return LEFT_COL_NOT_CORNER;
	if(coordinates[1] == dim_limit)
		return RIGHT_COL_NOT_CORNER;
	return -1; 
}
void initialize_randomly(int N, double **matrix, int rank){
	 srand(time(0)*(rank+1));
	 for(int i = 0; i< N; i++)
	 	for(int j = 0; j< N; j++)
	 		matrix[i][j] = (rank+1)*((double)rand()/ RAND_MAX);
}
double** alloc_2D_array_memory(int R, int C)
{
	double **matrix = (double**)malloc(R*sizeof(double*));
	for(int i = 0; i< R; i++)
		matrix[i] = (double*)malloc(C*sizeof(double));
	return matrix;
}

