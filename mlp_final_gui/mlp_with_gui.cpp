#include <vector>
#include <numeric>
#include <windows.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include<math.h>
using namespace std;
#include "TrainingItem.hpp"

vector<vector<double>> hidden_layer(10,vector<double>(10,0));  //#rows=hidden_nodes #columns=input_node
vector<vector<double>> output_layer(7,vector<double>(10,0));   //#rows=output_node  #columns=hidden_node
double output_delta[7];  
double hidden_delta[10];

void initialize_network(int input_n,int hidden_n,int output_n)   //initialize weight vector by radom value
{
	for(int i=0;i<hidden_n;i++)
		for(int j=0;j<(input_n+1);j++)
			hidden_layer[i][j]=((double) rand() / (RAND_MAX));
			
	for(int i=0;i<output_n;i++)
		for(int j=0;j<(hidden_n+1);j++)
			output_layer[i][j]=((double) rand() / (RAND_MAX));	
} 



double dot_product(const vector<double> &v1,const vector<double> &v2)
{
     return inner_product(v1.begin(), v1.end(), v2.begin(), ((double) rand() / (RAND_MAX)));
}
  
double transfer_deriavtive(double output)
{
	return(output*(1.0-output));
}


void back_propagate_error(const vector<double> &expected_output,const vector<double> &result_output,const vector<double> &hidden_output)
{
	double errors_o[7],errors_h[10];
	for(int i=0;i<7;i++)
	{
        errors_o[i]=(expected_output[i])-(result_output[i]);	
	}
	for(int i=0;i<7;i++)
	{
		output_delta[i]=errors_o[i]* transfer_deriavtive(result_output[i]);
	}
	for(int i=0;i<10;i++)
	{
		double error=0.0;
		for(int j=0;j<7;j++)
		{
			error+=output_layer[j][i]*output_delta[j];
		}
		errors_h[i]=error;
	}
	for(int i=0;i<10;i++)
	{
		hidden_delta[i]=errors_h[i]*transfer_deriavtive(hidden_output[i]);
	}	
		
}

void update_weight(const vector<double> &input,const vector<double> &hidden_input,double l_rate)
{
	for(int i=0;i<10;i++)
	{
		for(int j=0;j<10;j++)
		{
			hidden_layer[i][j]+=l_rate*input[j]*hidden_delta[i];
		}
	}
	for(int i=0;i<7;i++)
	{
		for(int j=0;j<10;j++)
		{
			output_layer[i][j]+=l_rate*hidden_input[j]*output_delta[i];
		}
	}
}
double sigmoid(double x)
{
	double exp_val;
	double ret_val;
	exp_val=exp((double)-x);
	ret_val=1/(1+exp_val);
	return ret_val;
}

double test_network(vector<TrainingItem> &test)
{
	double test_count=0;
	double correct_count=0;
	for (auto& item : test)
	{
		test_count++;
		vector<double> hidden_output;
			for(int j=0;j<10;j++)
			{
				hidden_output.push_back(sigmoid((dot_product(item.inputs(), hidden_layer[j]))));    //hidden_layer inputs generated
			}
			//cout<<"\nHidden layer output calculated";
			vector<double> result_output;
			for(int k=0;k<7;k++)
			{
				result_output.push_back(sigmoid((dot_product(hidden_output, output_layer[k]))));
			}
			double expOut=item.output();
			int max_ind=-1;
			double max=INT_MIN;
			for(int k=0;k<7;k++)
			{
				if(max<result_output[k])
				{
					max_ind=k+1;
					max=result_output[k];
				}
			}
			
			if(max_ind==expOut)
			{
				correct_count++;
				cout<<"\nCorrect Classified";
			}
			else
			{
				cout<<"\nIncorrect Classified";
			}
			
	}
	double accuracy=(correct_count/test_count)*100;
	return accuracy;
}

vector<double> train_network(vector<TrainingItem> &train,double l_rate,int n_epoch,int n_output)
{
	int i=0;
	vector<double> error;

	while(i<n_epoch)
	{
		cout<<"\nEpoch "<<i+1;
		double sum_error=0.0;
		
		for (auto& item : train)
		{
			vector<double> hidden_output;
			for(int j=0;j<10;j++)
			{
				hidden_output.push_back(sigmoid((dot_product(item.inputs(), hidden_layer[j]))));    //hidden_layer inputs generated
			}
			//cout<<"\nHidden layer output calculated";
			vector<double> result_output;
			for(int k=0;k<7;k++)
			{
				result_output.push_back(sigmoid((dot_product(hidden_output, output_layer[k]))));
			}
			//cout<<"\nFinal layer output calculated";
			vector<double> expected_output(7,0);
			double expOut=item.output();
			expected_output[expOut-1];
			double sum=0.0;
			for(int l=0;l<7;l++)
			{
				sum+=pow((expected_output[l]-result_output[l]),2);
			}
			sum_error+=sum;
			back_propagate_error(expected_output,result_output,hidden_output);
			update_weight(item.inputs(),hidden_output,l_rate);
		}
		error.push_back(sum_error/198);
		i++;
	}
	cout<<"\nModel trained";
	return error;
}
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance,LPSTR lpCmdLine, int nCmdShow)
{
	/*Importing dataset*/
	MessageBox(NULL, "Start training MLP", "Start", MB_OK);

	ifstream inFile;
	
     inFile.open("glass_identification1.csv");
     if (!inFile) 
	 {
    	cerr << "Unable to open file datafile.txt";
    	exit(1);   // call system to stop
	 }
	
	 
	 double a,b,c,d,e,f,g,h,i,j,k;
	 vector<TrainingItem> training_set;
	 while(!inFile.eof())
	 {
	 	inFile>>a>>b>>c>>d>>e>>f>>g>>h>>i>>j>>k;
	 	vector<double> input={a,b,c,d,e,f,g,h,i,j};
	 	TrainingItem temp(k,input);
	 	training_set.push_back(temp);
	 }
	 inFile.close();
	cout<<"\nDataset imported";
	/*Dataset imported*/
	
	vector<double> error;
	initialize_network(10,10,7);       //parameters are #inputs_nodes #hidden_nodes #output_nodes
	cout<<"\nNetwork initialized";
	
	error=train_network(training_set,0.5,50,7);
	cout<<"\nModel trained, back in main loop";
	ofstream outFile;
	outFile.open("error1.csv");
	if (!outFile) 
	 {
    	cerr << "Unable to open file datafile.txt";
    	exit(1);   // call system to stop
	 }
	 for(int i=0;i<20;i++)
	 {
	 	outFile<<i+1<<","<<error[i]<<"\n";
	 }
	 outFile.close();
	 
	 
	 MessageBox(NULL,"Start testing the model", "Test", MB_OK);
	 
	 ifstream in;
     in.open("test1.csv");
     if (!in) 
	 {
    	cerr << "Unable to open file datafile.txt";
    	exit(1);   // call system to stop
	 }

	 vector<TrainingItem> test_set;
	 while(!in.eof())
	 {
	 	in>>a>>b>>c>>d>>e>>f>>g>>h>>i>>j>>k;
	 	vector<double> input={a,b,c,d,e,f,g,h,i,j};
	 	TrainingItem temp(d,input);
	 	test_set.push_back(temp);
	 }
	 in.close();
	 double accuracy;
	 accuracy=test_network(test_set);
	 cout<<"\nAccuracy:"<<accuracy;
	 char acc[10];
	 sprintf(acc,"%lf",accuracy);
	 char s1[20]={"Accuracy:"};
	 strcat(s1,acc);
	 MessageBox(NULL, s1, "End", MB_OK);

	 return 0;
	
}
	
