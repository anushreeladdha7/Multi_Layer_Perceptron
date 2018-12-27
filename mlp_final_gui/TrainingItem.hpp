#ifndef TRAININGITEM_HPP
#define TRAININGITEM_HPP
using namespace std;
#include <vector>

class TrainingItem 
{
	private:
  			int output_;
  			vector<double> inputs_;
 	public:
  			TrainingItem(int expected_output, const vector<double> &inputs): output_(expected_output)
  			{
    			inputs_ = inputs;
 			}

  			int output() const { return output_; }
  			vector<double> & inputs() { return inputs_; }

 
};

#endif // TRAININGITEM_HPPs
