#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <unordered_map>
#include <string>
using namespace std;

struct Vector
{
	float *numeric_features;
	string label;
	Vector * next;

	~Vector()
	{
		if (numeric_features)
		{
			delete[] numeric_features;
		}
		if (next)
		{
			delete next;
		}
	}
};

string * Labels;

struct General_Summery
{

	int FeaturesCount = 0;
	int VectorsCount = 0;
	float *max_of_feature_by_index = nullptr;
	float *min_of_feature_by_index = nullptr;
	Vector * vectors;

	~General_Summery()
	{
		if (max_of_feature_by_index)
		{
			delete[] max_of_feature_by_index;
		}
		if (min_of_feature_by_index)
		{
			delete[] min_of_feature_by_index;
		}
		if (vectors)
		{
			delete vectors;
		}
	}
};

struct OutPut
{
	int FeaturesCount = 0;
	int VectorsCount = 0;
	Vector * vectors;

	~OutPut()
	{
		if (vectors)
		{
			delete vectors;
		}
	}
};

int count_width(ifstream & f)
{
	char ch;
	int count = 0;
	int indx = 0;

	while (true)
	{
		bool isDone = f.get(ch) ? false : true;

		//keep until not numeric
		if (ch == ',' || ch == ' ' || ch == '\n' || ch == '\0' || isDone)
		{
			if (indx > 0)
			{
				count++;
				indx = 0;
			}

			if (ch == '\n' || ch == '\0' || isDone)
			{
				if (isDone) f.clear();
				f.seekg(0, ios::beg);

				break;
			}

			continue;
		}

		indx++;
	}

	return count;
}

void print_vectors(Vector *first, int width)
{
	Vector *next = first;
	do {
		for (int i = 0; i < width; i++)
		{
			cout << next->numeric_features[i] << ':';
		}
		cout << next->label;
		cout << '\n';
	} while ((next = next->next));
}

void normalize_vectors(General_Summery & general_summery)
{
	Vector *vectors = general_summery.vectors;
	int width = general_summery.FeaturesCount;
	const Vector *next = vectors;
	do {
		for (int i = 0; i < width; i++)
		{
			float X = next->numeric_features[i];
			float min = general_summery.min_of_feature_by_index[i];
			float max = general_summery.max_of_feature_by_index[i];

			float Xnew = (X - min) / (max - min);
			next->numeric_features[i] = Xnew;
		}
	} while ((next = next->next));
}

void normalize_vec(General_Summery &general_summery, Vector *vec)
{
	int width = general_summery.FeaturesCount;

	for (int i = 0; i < width; i++)
	{
		float X = vec->numeric_features[i];
		float min = general_summery.min_of_feature_by_index[i];
		float max = general_summery.max_of_feature_by_index[i];

		float Xnew = (X - min) / (max - min);
		vec->numeric_features[i] = Xnew;
	}
}
void write_file(string path, General_Summery &model)
{

	ofstream sfile(path);

	//myfile.open(path,fstream::out);
	if (sfile.is_open())
	{
		Vector *next = model.vectors;
		int width = model.FeaturesCount;
		do { 	for (int i = 0; i < width; i++)
			{
				sfile << next->numeric_features[i] << ',';
			}
			sfile << next->label;
			sfile << '\n';	//There will be an empty line

		} while ((next = next->next));

		cout << "Done writing the file to " << path << "..." << '\n';
	}
	else cout << "WRITING FAILED" << '\n';

	sfile.close();
}
float minkowski_distance(float X[], float Y[], int width, float p)
{

	float result = 0;

	for (int i = 0; i < width; i++)
	{
		result += pow(abs(X[i] - Y[i]), p);
	}

	result = pow(result, p);

	return result;
}

float chipvoskie_distance(float X[], float Y[], int width)
{
	float max = -__FLT_MAX__;

	for (int i = 0; i < width; i++)
	{
		float result = abs(X[i] - Y[i]);
		if (result >= max)
		{
			max = result;
		}
	}

	return max;

}

float cosine_distance(float X[], float Y[], int width)
{

	float numerator = 0;

	for (int i = 0; i < width; i++)
	{
		numerator += X[i] *Y[i];
	}
	float denominator1 = 0;

	for (int i = 0; i < width; i++)
	{
		denominator1 += pow(X[i], 2);
	}

	float denominator2 = 0;
	for (int i = 0; i < width; i++)
	{
		denominator2 += pow(X[i], 2);
	}

	float denominator = sqrt(denominator1) *sqrt(denominator2);

	return numerator / denominator;
}

float ChiSqaure_Distance(float X[], float Y[], int width)
{
	float result = 0;
	for (int j = 0; j < width; j++)
	{
		float a = X[j] - Y[j];
		float b = X[j] + Y[j];
		if (b != 0)
			result += 0.5 *((a *a) / b);
	}

	return result;
}

float Gower_distance(float X[], float Y[], int width)
{
	float result = 0;
	for (int j = 0; j < width; j++)
	{

		float a = X[j] - Y[j];
		float b = X[j] + Y[j];
		if (b != 0)
			result += 0.5 *((a *a) / b);
	}

	return result;

}

struct K_node
{
	string label;
	float distance;
	K_node * next;
	K_node * pre;
};

//Distance of NNc(di)
float solve(float vector[], General_Summery &general_summery, int k, int choice, Vector **IndexVectors)
{
	//TODO expect k less than line number
	//Vector* vectors = general_summery.vectors;
	int width = general_summery.FeaturesCount;

	Vector *next = *IndexVectors;
	string targetClass = next->label;
	float mins_distances[k];
	fill_n(mins_distances, k, __FLT_MAX__);

	K_node *nex_k = new K_node();
	K_node *k_nodes = nex_k;

	int kcount = 0;
	for (int i = 1; i <= k; i++)
	{
		nex_k->distance = __FLT_MAX__;
		if (i != k)
		{
			nex_k->next = new K_node();
			nex_k = nex_k->next;
		}
		else
		{

			nex_k->next = NULL;
		}
	}
	do {
		if (!next) break;
		if (next->label != targetClass) break;

		float distance = minkowski_distance(vector, next->numeric_features, width, 2);

		K_node *node = k_nodes;
		K_node *new_node = k_nodes;
		K_node *second_node = k_nodes->next;
		K_node *previous = k_nodes;

		if (kcount < k) kcount++;
		new_node->distance = distance;
		new_node->label = next->label;
		while ((node = node->next))
		{

			//distance is smaller than the last/smaller k_node
			//must : k>1 .. by checking (node->pre)
			if (distance <= node->distance && !(node->next))
			{
				node->next = new_node;
				new_node->next = nullptr;
				//new_node->pre = node;
				k_nodes = second_node;
				//k_nodes->pre = nullptr;

				break;
			}
			else if (distance >= node->distance)
			{

				new_node->next = node;
				//incase of first element is targeted, it's already in the right position
				if (previous != k_nodes)
				{

					//new_node->pre = node->pre;
					//K_node* pre_node = node->pre;
					//pre_node->next = new_node;
					//node->pre = new_node;
					previous->next = new_node;
					k_nodes = second_node;
					k_nodes->pre = nullptr;
				}

				break;
			}

			previous = node;
		}
	} while ((next = next->next));

	while (next && (next->label == targetClass))
	{
		next = next->next;
	}

	*IndexVectors = next;	//save the last index

	//unordered_map<string, int> map;

	//K_node* node = k_nodes;

	return k_nodes->distance;
}

void consume_training_file(string path, General_Summery &general_summery)
{
	char ch;
	string str = "";

	ifstream myfile(path);

	if (myfile.is_open())
	{
		const int width = count_width(myfile) - 1;	//Last column is for label class
		general_summery.FeaturesCount = width;
		general_summery.max_of_feature_by_index = new float[width];
		general_summery.min_of_feature_by_index = new float[width];

		fill_n(general_summery.max_of_feature_by_index, width, __FLT_MIN__);
		fill_n(general_summery.min_of_feature_by_index, width, __FLT_MAX__);

		int indx = 0;

		Vector * vector;
		vector = new Vector();

		vector->numeric_features = new float[width];

		Vector *vectors = vector;
		bool isNumeric = true;
		Vector *previous = vector;

		while (true)
		{
			bool isDone = myfile.get(ch) ? false : true;

			//keep until not numeric
			if (ch == ',' || ch == ' ' || ch == '\n' || ch == '\0' || isDone)
			{
				if (str.length() > 0)
				{
					if (isNumeric && indx < width)
					{
						float val = stof(str);
						vector->numeric_features[indx] = val;
						if (val > general_summery.max_of_feature_by_index[indx])
						{
							general_summery.max_of_feature_by_index[indx] = val;
						}
						if (val < general_summery.min_of_feature_by_index[indx])
						{
							general_summery.min_of_feature_by_index[indx] = val;
						}

						indx++;
					}
					else if (indx == width)
					{
						//no need to check for the end of the line
						vector->label = str;
						general_summery.VectorsCount += 1;

						if (!isDone)
						{
							previous = vector;

							Vector *tmp = new Vector();
							vector->next = tmp;

							tmp->numeric_features = new float[width];

							vector = tmp;
						}
						else
						{
							vector->next = nullptr;
							break;
						}
						indx = 0;
					}
				}

				if (isDone)
				{
					if (indx == 0)
					{
						//There was empty lines
						delete vector;
						previous->next = NULL;
					}
					break;
				}

				isNumeric = true;	//reset for the next feature
				str = "";
				continue;
			}

			if (isNumeric) isNumeric = isNumeric && (isdigit(ch) || ch == '.');
			str += ch;
		}
		general_summery.vectors = vectors;
	}
	else
	{
		cout << "Unable to open file";
		exit(0);
	}

	myfile.close();

}

//TODO account for empty files
string solve_file(string path, General_Summery &model, General_Summery &general_summery, int k, int choice)
{
	char ch;
	string str = "";
	unordered_map<string, float> totals;

	//////////////////////////
	ifstream myfile(path);
	if (myfile.is_open())
	{

		const int width = model.FeaturesCount;
		general_summery.FeaturesCount = model.FeaturesCount;
		int indx = 0;

		Vector * vector;
		vector = new Vector();

		vector->numeric_features = new float[width];

		general_summery.vectors = vector;
		bool isNumeric = true;
		Vector *previous = vector;

		while (true)
		{

			bool isDone = myfile.get(ch) ? false : true;

			//keep until not numeric
			if (ch == ',' || ch == ' ' || ch == '\n' || ch == '\0' || isDone)
			{
				if (str.length() > 0)
				{
					if (isNumeric && indx < width)
					{
						float val = stof(str);
						vector->numeric_features[indx] = val;
						indx++;
					}

					if (isNumeric && indx == width)
					{
						//no need to check for the end of the line

						normalize_vec(model, vector);
						//NBNN
						Vector *vecPointer = model.vectors;
						while (vecPointer)
						{
							string label = vecPointer->label;
							float distance = solve(vector->numeric_features, model, k, choice, &vecPointer);
							if (totals.count(label))
							{
								totals[label] += distance;
							}
							else
							{
								totals[label] = distance;
							}
						}

						//NBNN

						general_summery.VectorsCount += 1;
						if (!isDone)
						{
							previous = vector;
							Vector *tmp = new Vector();
							vector->next = tmp;

							tmp->numeric_features = new float[width];

							vector = tmp;
						}

						indx = 0;
					}
				}
				if (isDone)
				{
					if (indx == 0)
					{
						delete vector;
						previous->next = NULL;
					}
					break;
				}

				str = "";
				continue;
			}
			if (isNumeric) isNumeric = isNumeric && (isdigit(ch) || ch == '.');

			str += ch;
		}
	}
	else cout << "Unable to open file";

	myfile.close();
	string result_class_label = "";
	int max = +__INT_MAX__;
	for (auto kv: totals)
	{
		cout << "For C = " << kv.first << '\n';
		cout << " Σc || di - NNC(di) ||2 = " << kv.second << '\n';

		if (max >= kv.second)
		{
			result_class_label = kv.first;
			max = kv.second;
		}
	}

	return result_class_label;

}

float accuracy(General_Summery &model1, General_Summery &model2)
{

	Vector *vectors1 = model1.vectors;
	Vector *vectors2 = model2.vectors;

	float correctCount = 0;
	do {

		if (vectors1->label == vectors2->label)
		{
			correctCount += 1.0;
		}
		if (!(vectors2 = vectors2->next)) break;
	} while ((vectors1 = vectors1->next));

	return ((float) correctCount / (float) model1.VectorsCount);
}
int main()
{

	General_Summery *model = new General_Summery;
	General_Summery *testData = new General_Summery;
	string trainPath;
	cout << '\n' << "Insert classefied Image Descriptors (directory path) : " << '\n';
	getline(std::cin, trainPath);

	consume_training_file(trainPath, *model);

	cout << '\n' << "DONE LOADING DATA..." << '\n' <<
		"Dimensions Count : " << model->FeaturesCount << '\n' <<
		"Image Describtors Count : " << model->VectorsCount << '\n' <<
		'\n';
	normalize_vectors(*model);
	cout << '\n' << "DONE Normalizing DATA..." << '\n';

	cout << '\n' << "Insert unclassefied Image Descriptors (Q) (directory path) : " << '\n';
	string unlabledPath;
	getline(cin, unlabledPath);

	cout << '\n';

	string result_class = solve_file(unlabledPath, *model, *testData, 1, 2);

	cout << '\n' << "Result..." << '\n' <<
		"The image Class (arg minC) is : " << result_class << '\n';

	//delete testData;
	//delete model;

	return 0;
}