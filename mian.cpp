#include <iostream>
#include <Eigen\Dense>

using namespace std;

typedef Eigen::Matrix<int, 3, 3> Matrix3i;
int main_Matrix()
{
	/*
	Matrix的初始化方法
	Eigen::Matrix<int, 3, 3> 
	int 代表Matrix的数据类型，3，3 分别代表 rows， cols
	Matrix3i m1;
	m1(0,0) = 1
	m1(0,1) = 2
	m1(0,2) = 3
	...

	或者用 m1 << 1,2,3 ...

	*/

	Matrix3i m1;
	m1 << 1, 2, 3, 4, 5, 6, 7, 8, 9;
	cout << "m1 = \n" << m1 << endl;

	Matrix3i m2;
	m2 << 1, 0, 0, 0, 1, 0, 0, 0, 1;
	cout << "m2 = \n" << m2 << endl;

	cout << "m1 * m2 = \n" << (m1 * m2) << endl;

	return 0;
}


#include "./eigen/unsupported/Eigen/CXX11/Tensor"

/*
Eigen 不同类型 tensor 的构造
*/
int Eigen_Construct()
{
	
	//一  Tensor 类

	/*
	Tensor 是模板类，模板中一共有4个参数，前三个参数的含义分别如下

	template<typename Scalar_, int NumIndices_, int Options_, typename IndexType_>
	class Tensor : public TensorBase<Tensor<Scalar_, NumIndices_, Options_, IndexType_> >
	1.float 代表Tensor 中存储的数据类型
	2.NumIndices_ 代表维度，即多维数组的维度，如3代表三维数组
	3. Options_ 可选参数，决定数据如何存储，如 Eigen::RowMajor

	
	*/
	//1.创建了一个3维的向量，明确了各个维度的尺寸分别是2，3，4，该向量分配了24个float 空间（24 = 2*3*4）
	Eigen::Tensor<float, 3, Eigen::RowMajor> t_3d(2, 3, 4);

	// 重新设置t_3d 向量的尺寸为（3，4，3），可以把他的不同维度设置不同的尺寸，但是维度个数需要一致
	t_3d = Eigen::Tensor<float, 3, Eigen::RowMajor>(3, 4, 3);

	//2.创建一个2维的向量，不明确各个维度的尺寸，而是通过数组的形式给出，如下面的维度用{5，7}数组给出

	Eigen::Tensor<string, 2> t_2d({ 5, 7 });



	//二 TensorFixedSize类
	/*
	TensorFixedSize<data_type, Sizes<size0, size1, ...>>

	template<typename Scalar_, typename Dimensions_, int Options_, typename IndexType>
	class TensorFixedSize : public TensorBase<TensorFixedSize<Scalar_, Dimensions_, Options_, IndexType> >
	1.float 代表Tensor 中存储的数据类型
	2.Dimensions_ 代表各个维度的尺寸
	3. Options_ 可选参数，决定数据如何存储，如 Eigen::RowMajor

	TensorFixedSize 需要在定义时明确各个维度的尺寸，因此运算速度较快
	*/

	//创建一个4*3 的 float 类型的 Tensor
	Eigen::TensorFixedSize<float, Eigen::Sizes<4, 3>> t_4x3;


	


	//三 TensorMap 类
	/*
	TensorMap<Tensor<data_type, rank>>


	template<typename PlainObjectType, int Options_, template <class> class MakePointer_> 
	class TensorMap : public TensorBase<TensorMap<PlainObjectType, Options_, MakePointer_> >
	1.PlainObjectType 
	2.Options_ 
	3. MakePointer_ 
	TensorMap用于在内存上创建一个张量，内存是由代码的另一部分分配和拥有的。它允许把任何一块分配的内存看作一个张量。
	此类的实例不拥有存储数据的内存。
	一句话总结：TensorMap 并不拥有内存，只是组织其他Tensor 。
	*/

	//可以通过传入一块儿内存，不同的维度构造TensorMap
	int storage[128];  // 2 x 4 x 2 x 8 = 128
	Eigen::TensorMap<Eigen::Tensor<int, 4>> t_4d(storage, 2, 4, 2, 8);

	//同一块儿内存可以被看作不同的TensorMap 
	Eigen::TensorMap<Eigen::Tensor<int, 2>> t_2d_2(storage, 16, 8);

	Eigen::TensorFixedSize<float, Eigen::Sizes<4, 3>> t_4x3_2;
	Eigen::TensorMap<Eigen::Tensor<float, 1>> t_12(t_4x3.data(), 12);


	return 1;
}

void visitTensorElement()
{
	/*
	1. 通过指定不同的下标来访问元素 
	tensorName(index0, index1...)
	*/

	Eigen::Tensor<float, 3> t_3d(2, 3, 4);
	t_3d(0, 1, 0) = 12.0f;

	// Initialize all elements to random values.
	for (int i = 0; i < 2; ++i) 
	{
		for (int j = 0; j < 3; ++j) 
		{
			for (int k = 0; k < 4; ++k) 
			{
				t_3d(i, j, k) = rand();
				cout << t_3d(i, j, k) << " ";
			}
			cout << endl;
		}
		cout << endl;
	}


	// Print elements of a tensor.
	for (int i = 0; i < 2; ++i) {
		cout << t_3d(i, 0, 0) << endl;
	}


}

void tensorLayout()
{
	//如下面分别设置了列优先  和 行优先
	//Eigen::Tensor<float, 3, Eigen::ColMajor> col_major1;  // equivalent to Tensor<float, 3>
	//float storage[128];  // 2 x 4 x 2 x 8 = 128
	//Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::ColMajor> > row_major1(storage, 2, 2, 4, 8);


	//
	Eigen::Tensor<float, 2, Eigen::ColMajor> col_major(2, 4);
	Eigen::Tensor<float, 2, Eigen::RowMajor> row_major(2, 4);

	Eigen::Tensor<float, 2> col_major_result = col_major;  // 默认为colMajor ，layout方式相同，因此可以赋值
	//Eigen::Tensor<float, 2> col_major_result2 = row_major;  // layout方式不同，编译出错，错误信息为  error C2338:  YOU_MADE_A_PROGRAMMING_MISTAKE

	// Simple layout swap
	col_major_result = row_major.swap_layout();
	eigen_assert(col_major_result.dimension(0) == 4);
	eigen_assert(col_major_result.dimension(1) == 2);

}

void tensorCompute()
{
	Eigen::Tensor<int, 2> t1(2, 2);
	Eigen::Tensor<int, 2> t2(2, 2);
	for (int i = 0;i < 2; i++)
		for (int j = 0; j < 2; j++)
		{
			t1(i, j) = 1;
			t2(i, j) = 2;
		}

	Eigen::Tensor<int, 2> t3 = t1 + t2;
	cout << "t1:" << endl << t1 << endl;
	cout << "t2:" << endl << t2 << endl;
	
	cout << "t3:" << endl << t3 << endl;


	auto t4 = t3 * 2;           // t4 is an Operation.
	Eigen::Tensor<int, 2> result = t4;  // The operations are evaluated
	cout << t4 << endl;


}

void tensorCompute2()
{
	// The previous example could have been written:
	Eigen::Tensor<float, 2> t1(2, 2);
	Eigen::Tensor<float, 2> t2(2, 2);
	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 2; j++)
		{
			t1(i, j) = 1.0f;
			t2(i, j) = 2.0f;
		}

	Eigen::Tensor<float, 2> result = ((t1 + t2) * 0.2f).exp();

	// If you want to compute (t1 + t2) once ahead of time you can write:
	Eigen::Tensor<float, 2> result2 = ((t1 + t2).eval() * 0.2f).exp();
	cout << result2 << endl;


	Eigen::TensorRef<Eigen::Tensor<float, 2> > ref = ((t1 + t2) * 0.2f).exp();

	// Use "ref" to access individual elements.  The expression is evaluated
	// on the fly.
	float at_0 = ref(0, 0, 0);
	cout << ref(0, 1, 0);

}

void testTensorInfo()
{
	//1. NumDimensions 输出维度个数
	Eigen::Tensor<float, 2> a(3, 4);
	cout << "Dims " << a.NumDimensions <<endl; // 输出：Dims:2

	//2. dimensions() 输出不同维度的size
	//typedef DSizes<Index, NumIndices_> Dimensions;
	const Eigen::Tensor<float, 2>::Dimensions & d = a.dimensions();

	cout << "Dim size: " << d.size()<< ", dim 0: " << d[0]
		<< ", dim 1: " << d[1] <<endl; //Dim size: 2, dim 0: 3, dim 1: 4

	//3. Index dimension(Index n) 输出第n 维的维度
	int dim1 = a.dimension(1);
	cout << "Dim 1: " << dim1 <<endl; // Dim 1: 4

	//4. Index size() 输出tensor的总元素个数

	cout << "Size: " << a.size() <<endl;

}

void testInitializer()
{
	//1. 内容初始化
	/*
	当Tensor 或TensorFixedSize 创建的时候，已经分配好了内存空间，但是内存并没有初始化，所以这里给出了初始化方法
	*/
	Eigen::Tensor<float, 2> a(3, 4);

	a.setConstant(12.3f);
	cout << "Constant: " << endl << a << endl << endl;


	Eigen::Tensor<string, 2> b(2, 3);
	b.setConstant("yolo");
	cout << "String tensor: " << endl << b << endl << endl;

	a.setZero();
	cout << "Zero Constant: " << endl << a << endl << endl;


	Eigen::Tensor<float, 2> c(2, 3);
	c.setValues({ {0.0f, 1.0f, 2.0f}, {3.0f, 4.0f, 5.0f} });
	cout << "c setValues" << endl << c << endl << endl;


	Eigen::Tensor<int, 2> d(2, 3);
	d.setConstant(1000);
	d.setValues({ {10, 20, 30} });
	cout << "d" << endl << d << endl << endl;

	c.setRandom();
	cout << "Random: " << endl << c << endl << endl;

}
void testAccess()
{
	Eigen::Tensor<float, 2> c(2, 3);
	c.setValues({ {0.0f, 1.0f, 2.0f}, {3.0f, 4.0f, 5.0f} });
	cout << c(0,1) <<endl;
	cout << c(0) << endl;

	Eigen::Tensor<float, 2> a(3, 4);
	float* a_data = a.data();
	a_data[0] = 123.45f;
	cout << "a(0, 0): " << a(0, 0);

}

void testOperation()
{
	Eigen::Tensor<float, 2> a(2, 3);
	a.setConstant(1.0f);
	Eigen::Tensor<float, 2> b = a + a.constant(2.0f);
	Eigen::Tensor<float, 2> c = b * b.constant(0.2f);

	cout << "a" << endl << a << endl << endl;
	cout << "b" << endl << b << endl << endl;
	cout << "c" << endl << c << endl << endl;

	Eigen::Tensor<float, 2> d = c * c.random();
	cout << "d" << endl << d << endl << endl;
}

void testUnary()
{
	Eigen::Tensor<int, 2> a(2, 3);
	a.setValues({ {0, 1, 8}, {27, 64, 125} });
	Eigen::Tensor<double, 2> b = a.cast<double>().pow(1.0 / 3.0);
	Eigen::Tensor<double, 2> sqrt = a.cast<double>().sqrt();
	Eigen::Tensor<double, 2> rsqrt = a.cast<double>().rsqrt();
	Eigen::Tensor<double, 2> square = a.cast<double>().square();
	Eigen::Tensor<double, 2> inverse = a.cast<double>().inverse();
	Eigen::Tensor<double, 2> exp = a.cast<double>().exp();
	Eigen::Tensor<double, 2> log = a.cast<double>().log();
	Eigen::Tensor<double, 2> abs = a.cast<double>().abs();
	Eigen::Tensor<int, 2> multiply = a * 2;
	cout << "a" << endl << a << endl << endl;
	cout << "b" << endl << b << endl << endl;

	cout << "b" << endl << b << endl << endl;
	cout << "sqrt" << endl << sqrt << endl << endl;
	cout << "rsqrt" << endl << rsqrt << endl << endl;
	cout << "square" << endl << square << endl << endl;
	cout << "inverse" << endl << inverse << endl << endl;
	cout << "exp" << endl << exp << endl << endl;
	cout << "log" << endl << log << endl << endl;
	cout << "abs" << endl << abs << endl << endl;
	cout << "multiply" << endl << multiply << endl << endl;


	Eigen::Tensor<float, 2> f(2, 3);
	f.setConstant(1.0f);
	Eigen::Tensor<float, 2> g = -f;
	cout << "f" << endl << f << endl << endl;
	cout << "g" << endl << g << endl << endl;

}


void testBinary()
{
	Eigen::Tensor<int, 2> a(2, 3);
	a.setValues({ {0, 1, 8}, {27, 64, 125} });

	Eigen::Tensor<int, 2> b = a * 3;

	cout << "a" << endl << a << endl << endl;
	cout << "b" << endl << b << endl << endl;
	cout << "a+b" << endl << a + b << endl << endl;
	cout << "a-b" << endl << a - b << endl << endl;
	cout << "a*b" << endl << a * b << endl << endl;
	cout << "a.cwiseMax(b)" << endl <<a.cwiseMax(b) << endl << endl;
	cout << "b.cwiseMax(a)" << endl << b.cwiseMax(a) << endl << endl;
	cout << "a.cwiseMin(b)" << endl << a.cwiseMin(b) << endl << endl;
	cout << "b.cwiseMin(a)" << endl << b.cwiseMin(a) << endl << endl;

}

void testSelection()
{
	Eigen::Tensor<bool, 2> _if(2,2);
	_if.setValues({ { false,true }, { true,false } });
	Eigen::Tensor<int, 2> _then (2,2);
	_then.setValues({ { 2,2 }, { 2,2 } });
	Eigen::Tensor<int, 2> _else (2,2);
	_else.setValues({ { 10,10 }, { 10,10 } });
	Eigen::Tensor<int, 2> result = _if.select(_then, _else);
	cout << "result:" << endl << result << endl;

}

void testReduction()
{
	// Create a tensor of 2 dimensions
	Eigen::Tensor<int, 2> a(2, 3);
	a.setValues({ {1, 2, 3}, {6, 5, 4} });
	// 
	Eigen::array<int, 1> dims =  { 1 }; //沿着第二个维度降维
	Eigen::array<int, 1> dims2 = { 0 }; //沿着第一个维度降维
	// maximum 返回的是某个维度的最大值

	Eigen::Tensor<int, 1> b = a.maximum(dims);
	cout << "a" << endl << a << endl << endl;
	cout << "b" << endl << b << endl << endl;


	Eigen::Tensor<int, 1> c = a.maximum(dims2);
	cout << "c" << endl << c << endl << endl;
}

void testReduction2()
{
	Eigen::Tensor<float, 3, Eigen::ColMajor> a(2, 3, 4);
	a.setValues({ {{0.0f, 1.0f, 2.0f, 3.0f},
				  {7.0f, 6.0f, 5.0f, 4.0f},
				  {8.0f, 9.0f, 10.0f, 11.0f}},
				 {{12.0f, 13.0f, 14.0f, 15.0f},
				  {19.0f, 18.0f, 17.0f, 16.0f},
				  {20.0f, 21.0f, 22.0f, 23.0f}} });
	//a有三个维度，我们沿着前两个维度降维，降维的结果是一个一维的Tensor,
	
	Eigen::Tensor<float, 1, Eigen::ColMajor> b =a.maximum(Eigen::array<int, 2>({ 0, 1 }));
	cout << "b" << endl << b << endl << endl;
}

void testReduction3()
{
	Eigen::Tensor<float, 3> a(2, 3, 4);
	a.setValues({ {{0.0f, 1.0f, 2.0f, 3.0f},
				  {7.0f, 6.0f, 5.0f, 4.0f},
				  {8.0f, 9.0f, 10.0f, 11.0f}},
				 {{12.0f, 13.0f, 14.0f, 15.0f},
				  {19.0f, 18.0f, 17.0f, 16.0f},
				  {20.0f, 21.0f, 22.0f, 23.0f}} });
	cout << "a:" << endl << a << endl << endl;
	Eigen::Tensor<float, 0> b = a.sum();
	cout << "b" << endl << b << endl << endl;
}




void testReduction4()
{
	// Create a tensor of 2 dimensions
	Eigen::Tensor<int, 2> a(3, 3);
	a.setValues({ {1, 2, 3}, {6, 5, 4},{8, 9, 10} });
	// 
	Eigen::array<int, 1> dims = { 1 }; //沿着第二个维度降维
	Eigen::array<int, 1> dims2 = { 0 }; //沿着第一个维度降维

	Eigen::Tensor<int, 1> maximum = a.maximum(dims);
	cout << "a" << endl << a << endl << endl;

	cout << "maximum(dims):" << endl << a.maximum(dims) << endl << endl;
	cout << "maximum():" << endl << a.maximum() << endl << endl;

	cout << "sum(dims):" << endl << a.sum(dims) << endl << endl;
	cout << "sum():" << endl << a.sum() << endl << endl;

	cout << "mean(dims):" << endl << a.mean(dims) << endl << endl;
	cout << "mean():" << endl << a.mean() << endl << endl;

	cout << "minimum(dims):" << endl << a.minimum(dims) << endl << endl;
	cout << "minimum():" << endl << a.minimum() << endl << endl;
	//返回相应维度元素的乘积
	cout << "prod(dims):" << endl << a.prod(dims) << endl << endl;
	cout << "prod():" << endl << a.prod() << endl << endl;
	//如果相应维度的元素都是大于0 ，则相应维度的降维结果为1 ，否则为0
	cout << "all(dims):" << endl << a.all(dims) << endl << endl;
	cout << "all():" << endl << a.all() << endl << endl;
	//如果相应维度的元素某个大于0 ，则相应维度的降维结果为1 ，否则为0
	cout << "any(dims):" << endl << a.any(dims) << endl << endl;
	cout << "any():" << endl << a.any() << endl << endl;


}


void testScan()
{
	// Create a tensor of 2 dimensions
	Eigen::Tensor<int, 2> a(2, 3);
	a.setValues({ {1, 2, 3}, {4, 5, 6} });
	// Scan it along the second dimension (1) using summation
	Eigen::Tensor<int, 2> b = a.cumsum(1);
	Eigen::Tensor<int, 2> c = a.cumprod(1);
	// The result is a tensor with the same size as the input
	cout << "a" << endl << a << endl << endl;
	cout << "cumsum" << endl << b << endl << endl;
	cout << "cumpord" << endl << c << endl << endl;
}


void testConvolve()
{
	Eigen::Tensor<float, 4, Eigen::RowMajor> input(3, 3, 7, 11);
	Eigen::Tensor<float, 2, Eigen::RowMajor> kernel(2, 2);
	Eigen::Tensor<float, 4, Eigen::RowMajor> output(3, 2, 6, 11);
    input.setRandom();
	kernel.setRandom();

	Eigen::array<int, 2> dims= { 1, 2 };  // Specify second and third dimension for convolution.
	output = input.convolve(kernel, dims);
	cout << "Kernel:" << endl << kernel << endl;
	cout << "Output:" << endl << output << endl;
	//下面手工计算卷积，对比结果
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 2; ++j) {
			for (int k = 0; k < 6; ++k) {
				for (int l = 0; l < 11; ++l) {
					const float result = output(i, j, k, l);
					const float expected = input(i, j + 0, k + 0, l) * kernel(0, 0) +
						input(i, j + 1, k + 0, l) * kernel(1, 0) +
						input(i, j + 0, k + 1, l) * kernel(0, 1) +
						input(i, j + 1, k + 1, l) * kernel(1, 1);
					cout << result << "," << expected << endl;
				}
			}
		}
	}
}


void testReshape()
{
	Eigen::Tensor<float, 2, Eigen::ColMajor> a(2, 3);
	a.setValues({ {0.0f, 100.0f, 200.0f}, {300.0f, 400.0f, 500.0f} });
	Eigen::array<Eigen::DenseIndex, 1> one_dim = { 3 * 2 };
	Eigen::Tensor<float, 1, Eigen::ColMajor> b = a.reshape(one_dim);
	array<Eigen::DenseIndex, 3> three_dims = { {3, 2, 1} };
	Eigen::Tensor<float, 3, Eigen::ColMajor> c = a.reshape(three_dims);
	cout << "a" << endl << a << endl;
	cout << "b" << endl << b << endl;
	cout << "c" << endl << c << endl;
}


void testShuffle()
{

	Eigen::Tensor<float, 3> input(2, 3, 3);
	input.setRandom();
	Eigen::array<Eigen::DenseIndex, 3> shuffle = { 1, 2, 0 };
	Eigen::Tensor<float, 3> output = input.shuffle(shuffle);
	cout << "input:" << endl <<  input << endl;
	cout << "output:" << endl << output << endl;
	cout << (output.dimension(0) == 3) <<endl;
	cout << (output.dimension(1) == 3) << endl;
	cout << (output.dimension(2) == 2) << endl;
}

void testStrides()
{
	Eigen::Tensor<int, 2> a(4, 3);
	a.setValues({ {0, 100, 200}, {300, 400, 500}, {600, 700, 800}, {900, 1000, 1100} });
	Eigen::array<Eigen::DenseIndex, 2> strides = { 3, 2 };
	Eigen::Tensor<int, 2> b = a.stride(strides);
	cout << "a" << endl << a << endl;
	cout << "b" << endl << b << endl;
}

void testSlice()
{
	Eigen::Tensor<int, 2> a(4, 3);
	a.setValues({ {0, 100, 200}, {300, 400, 500},
				 {600, 700, 800}, {900, 1000, 1100} });
	Eigen::array<Eigen::DenseIndex, 2> offsets = { 1, 0 };
	Eigen::array<Eigen::DenseIndex, 2> extents = { 2, 2 };
	Eigen::Tensor<int, 2> slice = a.slice(offsets, extents);
	cout << "a" << endl << a << endl;
	cout << "slice:" << endl << slice << endl;
}

void testChip()
{
	Eigen::Tensor<int, 2> a(4, 3);
	a.setValues({ {0, 100, 200}, {300, 400, 500},
				 {600, 700, 800}, {900, 1000, 1100} });
	Eigen::Tensor<int, 1> row_3 = a.chip(2, 0);
	Eigen::Tensor<int, 1> col_2 = a.chip(1, 1);
	cout << "a" << endl << a << endl;
	cout << "row_3" << endl << row_3 << endl;
	cout << "col_2" << endl << col_2 << endl;
}

void testReserve()
{
	Eigen::Tensor<int, 2> a(4, 3);
	a.setValues({ {0, 100, 200}, {300, 400, 500},
				{600, 700, 800}, {900, 1000, 1100} });
	Eigen::array<bool, 2> reverse = { true, false }; //表示第一维反转，第二维不反转
	Eigen::Tensor<int, 2> b = a.reverse(reverse);
	cout << "a" << endl << a << endl << "b" << endl << b << endl;
}
int main()
{
	testSlice();
}