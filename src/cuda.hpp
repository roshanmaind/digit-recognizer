bool cuda_init(void);
void cuda_destroy(void);

template <typename T>
void cuda_allocate(T** ptr, long long size);
template <typename T>
void cuda_delete(T** ptr);

void cuda_copy_to_device(float**, float**, int);

void cuda_matrix_mul(float**, float**, float**, int, int, int);
void cuda_matrix_add(float**, float**, float**, int, int);
void cuda_ReLU(float**, float**, int);
void cuda_softmax(float**, float**, int);
void cuda_cross_entropy_prime(float**, float**, float**, int);
void cuda_softmax_prime(float**, float**, float**, int);
void cuda_ReLU_prime_biases(float**, float**, float**, int);
void cuda_ReLU_prime_others(float**, float**, float**, float**, float**, float**, int, int);
void cuda_gradient_descent_step(float**, float**, float, int, int);