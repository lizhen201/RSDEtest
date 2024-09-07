/*#Paper:  Differential Evolution with Ring Sub-Population Architecture for Optimization
All the code of the RSDE is contained in "RSDE.cpp" file.
Compilation is simple using gcc/g++:
g++ RSDE.cpp -o RSDE.exe -std=c++11 -O3 -march=corei7-avx -fexpensive-optimizations -fomit-frame-pointer
Please note that the compilation requires support of C++11 standard. 
You may omit everything after "-O3", however, these options give a significant boost on most systems.
This will create RSDE.exe, available for running.
Next, the main optimization loop will be started, writing data to "RSDE_(F)_(DIM).txt", 
where F and DIM are the function number and problem dimension. */
#include <cmath>
#include <time.h>
#include <iomanip>
#include <vector>
#include <cstring>
#include <sstream>
#include <fstream>
#include <iostream>
#include <random>
#include <algorithm>

#include "cec17_test_func.cpp"

using namespace std;
unsigned globalseed = unsigned(time(NULL));
//unsigned globalseed = 2018;
unsigned seed1 = globalseed + 0;
unsigned seed2 = globalseed + 100;
unsigned seed3 = globalseed + 200;
unsigned seed4 = globalseed + 300;
unsigned seed5 = globalseed + 400;
std::mt19937 generator_uni_i(seed1);
std::mt19937 generator_uni_r(seed2);
std::mt19937 generator_norm(seed3);
std::mt19937 generator_cachy(seed4);
std::mt19937 generator_uni_i_2(seed5);
std::uniform_int_distribution<int> uni_int(0, 32768);
std::uniform_real_distribution<double> uni_real(0.0, 1.0);
std::normal_distribution<double> norm_dist(0.0, 1.0);
std::cauchy_distribution<double> cachy_dist(0.0, 1.0);
double jumping_rate = 0.2;

int IntRandom(int target) {
	if (target == 0)
		return 0;
	return uni_int(generator_uni_i) % target;
}
double Random(double minimal, double maximal) {
	return uni_real(generator_uni_r) * (maximal - minimal) + minimal;
}
double NormRand(double mu, double sigma) {
	return norm_dist(generator_norm) * sigma + mu;
}
double CachyRand(double mu, double sigma) {
	return cachy_dist(generator_cachy) * sigma + mu;
}
void qSort2int(double* Mass, int* Mass2, int low, int high) {
	int i = low;
	int j = high;
	double x = Mass[(low + high) >> 1];
	do {
		while (Mass[i] < x) ++i;
		while (Mass[j] > x) --j;
		if (i <= j) {
			double temp = Mass[i];
			Mass[i] = Mass[j];
			Mass[j] = temp;
			double temp2 = Mass2[i];
			Mass2[i] = Mass2[j];
			Mass2[j] = temp2;
			i++;
			j--;
		}
	} while (i <= j);
	if (low < j) qSort2int(Mass, Mass2, low, j);
	if (i < high) qSort2int(Mass, Mass2, i, high);
}

void cec17_test_func(double*, double*, int, int, int);
double* OShift, * M, * y, * z, * x_bound;
int ini_flag = 0, n_flag, func_flag, * SS;
int GNVars;
double ResultsArray[51][14];
int LastFEcount;
int NFEval = 0;
int MaxFEval = 0;
double tempF[1];
double fopt[1];
char buffer[30];
double globalbest;
bool globalbestinit;
bool initfinished;
vector<double> FitTemp;

void GetOptimum(int func_num, double* xopt, double* fopt) {
	FILE* fpt;
	char FileName[30];
	sprintf(FileName, "input_data/shift_data_%d.txt", func_num);
	fpt = fopen(FileName, "r");
	if (fpt == NULL)
		printf("\n Error: Cannot open input file for reading \n");
	for (int k = 0; k < GNVars; k++)
		fscanf(fpt, "%lf", &xopt[k]);
	fclose(fpt);
	cec17_test_func(xopt, fopt, GNVars, 1, func_num);
}

void FindLimits(double* Ind, double* Parent, int CurNVars, double CurLeft, double CurRight) {
	for (int j = 0; j < CurNVars; j++) {
		if (Ind[j] < CurLeft)
			Ind[j] = (CurLeft + Parent[j]) / 2.0;
		if (Ind[j] > CurRight)
			Ind[j] = (CurRight + Parent[j]) / 2.0;
	}
}
class Optimizer {
public:
	bool FitNotCalculated;
	double F;
	double F2;
	double Cr;
	int Int_ArchiveSizeParam;
	int MemorySize;
	int MemoryIter;
	int SuccessFilled;
	int MemoryCurrentIndex;

	int NVars;
	int NInds;
	int NIndsMax;
	int NIndsMin;

	double bestfit;
	int besti;

	int func_num;
	int TheChosenOne;
	int Generation;
	double ArchiveSizeParam;
	double psize;
	double psizeParam;
	int ArchiveSize;
	int CurrentArchiveSize;
	double ArchiveProb;

	double* Donor;
	double* Trial;
	int* Rands;

	double** Popul;
	double** PopulTemp;
	double* FitMass;
	double* FitMassTemp;
	double* FitMassCopy;
	double* BestInd;
	int* Indexes;
	double** Archive;
	double* FitMassArch;

	double* tempSuccessCr;
	double* tempSuccessF;

	double Right;
	double Left;

	double* MemoryCr;
	double* MemoryF;
	double* FitDelta;

	void Initialize(int newNInds, int newNVars, int func_num,
		double NewArchSizeParam, double NewArchiveProbParam, double NewPSize);
	void Clean();
	void MainCycle(ofstream& outFile);
	void FindNSaveBest(bool init);

	void PSO_MoveP();
	void PSO_UpdateBests(int TheChosenOne);

	void CopyToArchive(double* RefusedParent, double RefusedFitness);
	void SaveSuccessCrF(double Cr, double F, double FitD);
	void UpdateMemoryCrF();
	double MeanWL(double* Vector, double* TempWeights, int Size);
	void RemoveWorst(int NInds, int NewNInds);

	int GetImprState();
	bool GetFitState();

	int SelectWorst();
	int SelectBest();
};
double cec_17_(double* HostVector, int func_num) {
	cec17_test_func(HostVector, tempF, GNVars, 1, func_num);
	NFEval++;
	return tempF[0];
}
void Optimizer::Initialize(int newNInds, int newNVars, int newfunc_num,
	double NewArchSizeParam, double NewArchiveProbParam, double NewPSize) {
	FitNotCalculated = true;
	NInds = newNInds;
	NIndsMax = NInds;
	NIndsMin = 4;
	NVars = newNVars;
	Left = -100;
	Right = 100;
	Cr = 0.5;
	F = 0.8;
	besti = 0;
	Generation = 0;
	TheChosenOne = 0;
	CurrentArchiveSize = 0;
	psizeParam = NewPSize;
	ArchiveSizeParam = NewArchSizeParam;
	Int_ArchiveSizeParam = ceil(ArchiveSizeParam);
	ArchiveSize = NIndsMax * ArchiveSizeParam;
	ArchiveProb = NewArchiveProbParam;
	func_num = newfunc_num;

	Popul = new double* [NIndsMax];
	for (int i = 0; i != NIndsMax; i++)
		Popul[i] = new double[NVars];
	PopulTemp = new double* [NIndsMax];
	for (int i = 0; i != NIndsMax; i++)
		PopulTemp[i] = new double[NVars];
	Archive = new double* [NIndsMax * Int_ArchiveSizeParam];
	for (int i = 0; i != NIndsMax * Int_ArchiveSizeParam; i++)
		Archive[i] = new double[NVars];
	FitMassArch = new double[NIndsMax * Int_ArchiveSizeParam];
	FitMass = new double[NIndsMax];
	FitMassTemp = new double[NIndsMax];
	FitMassCopy = new double[NIndsMax];
	Indexes = new int[NIndsMax];
	BestInd = new double[NVars];

	for (int i = 0; i < NIndsMax; i++)
		for (int j = 0; j < NVars; j++)
			Popul[i][j] = Random(Left, Right);

	Donor = new double[NVars];
	Trial = new double[NVars];
	Rands = new int[NIndsMax];

	tempSuccessCr = new double[NIndsMax];
	tempSuccessF = new double[NIndsMax];
	FitDelta = new double[NIndsMax];

	for (int i = 0; i != NIndsMax; i++) {
		tempSuccessCr[i] = 0;
		tempSuccessF[i] = 0;
	}

	MemorySize = 5;
	MemoryIter = 0;
	SuccessFilled = 0;

	MemoryCr = new double[MemorySize];
	MemoryF = new double[MemorySize];
	for (int i = 0; i != MemorySize; i++) {
		MemoryCr[i] = 0.8 + 0.0 * Random(0, 1);
		MemoryF[i] = 0.3 + 0.0 * Random(0, 1);
	}
}
void Optimizer::SaveSuccessCrF(double Cr, double F, double FitD) {
	tempSuccessCr[SuccessFilled] = Cr;
	tempSuccessF[SuccessFilled] = F;
	FitDelta[SuccessFilled] = FitD;
	SuccessFilled++;
}
void Optimizer::UpdateMemoryCrF() {
	if (SuccessFilled != 0) {
		double Old_F = MemoryF[MemoryIter];
		double Old_Cr = MemoryCr[MemoryIter];
		double tempmax = tempSuccessCr[0];
		for (int i = 0; i != SuccessFilled; i++)
			if (tempSuccessCr[i] > tempmax)
				tempmax = tempSuccessCr[i];
		if (MemoryCr[MemoryIter] == -1 || tempmax == 0)
			MemoryCr[MemoryIter] = -1;
		else
			MemoryCr[MemoryIter] = (MeanWL(tempSuccessCr, FitDelta, SuccessFilled) + Old_Cr) / 2.0;  // strategy  D
			MemoryF[MemoryIter] = (MeanWL(tempSuccessF, FitDelta, SuccessFilled) + Old_F) / 2.0;
		MemoryIter++;
		if (MemoryIter >= MemorySize)
			MemoryIter = 0;
	}
}
double Optimizer::MeanWL(double* Vector, double* TempWeights, int Size) {
	double SumWeight = 0;
	double SumSquare = 0;
	double Sum = 0;
	for (int i = 0; i != SuccessFilled; i++)
		SumWeight += TempWeights[i];
	double* Weights = new double[SuccessFilled];

	for (int i = 0; i != SuccessFilled; i++)
		Weights[i] = TempWeights[i] / SumWeight;
	for (int i = 0; i != SuccessFilled; i++)
		SumSquare += Weights[i] * Vector[i] * Vector[i];
	for (int i = 0; i != SuccessFilled; i++)
		Sum += Weights[i] * Vector[i];
	delete Weights;
	if (fabs(Sum) > 0.000001)
		return SumSquare / Sum;
	else
		return 0.5;
}
void Optimizer::CopyToArchive(double* RefusedParent, double RefusedFitness) {
	if (CurrentArchiveSize < ArchiveSize) {
		for (int i = 0; i != NVars; i++)
			Archive[CurrentArchiveSize][i] = RefusedParent[i];
		FitMassArch[CurrentArchiveSize] = RefusedFitness;
		CurrentArchiveSize++;
	}
	else {
		int RandomNum = IntRandom(ArchiveSize);
		for (int i = 0; i != NVars; i++)
			Archive[RandomNum][i] = RefusedParent[i];
		FitMassArch[RandomNum] = RefusedFitness;
	}
}
void Optimizer::FindNSaveBest(bool init) {
	if (FitMass[TheChosenOne] <= bestfit || init) {
		bestfit = FitMass[TheChosenOne];
		besti = TheChosenOne;
		for (int j = 0; j != NVars; j++)
			BestInd[j] = Popul[besti][j];
	}
	if (bestfit < globalbest)
		globalbest = bestfit;
}
void Optimizer::RemoveWorst(int NInds, int NewNInds) {
	int PointsToRemove = NInds - NewNInds;
	for (int L = 0; L != PointsToRemove; L++) {
		double WorstFit = FitMass[0];
		int WorstNum = 0;
		for (int i = 1; i != NInds; i++) {
			if (FitMass[i] > WorstFit) {
				WorstFit = FitMass[i];
				WorstNum = i;
			}
		}
		for (int i = WorstNum; i != NInds - 1; i++) {
			for (int j = 0; j != NVars; j++)
				Popul[i][j] = Popul[i + 1][j];
			FitMass[i] = FitMass[i + 1];
		}
	}
}
void Optimizer::MainCycle(ofstream& outFile) {
	for (int TheChosenOne = 0; TheChosenOne != NInds; TheChosenOne++)
	{
		FitMass[TheChosenOne] = cec_17_(Popul[TheChosenOne], func_num);
		FindNSaveBest(TheChosenOne == 0);
		if (!globalbestinit || bestfit < globalbest) {
			globalbest = bestfit;
			globalbestinit = true;
		}
		if (NFEval % 100 == 0) {
			double temp = globalbest - fopt[0];
			if (temp <= 10E-8)
				temp = 0;
			outFile << temp << ",";
		}
	}
	do {
		double minfit = FitMass[0];
		double maxfit = FitMass[0];
		int prand;
		minfit = FitMass[0];
		maxfit = FitMass[0];
		for (int i = 0; i != NInds; i++) {
			FitMassCopy[i] = FitMass[i];
			Indexes[i] = i;
			if (FitMass[i] >= maxfit)
				maxfit = FitMass[i];
			if (FitMass[i] <= minfit)
				minfit = FitMass[i];
		}
		if (minfit != maxfit)
			qSort2int(FitMassCopy, Indexes, 0, NInds - 1);
		FitTemp.resize(NInds);
		for (int i = 0; i != NInds; i++)
			FitTemp[i] = 3.0 * (NInds - i);
		std::discrete_distribution<int> ComponentSelector(FitTemp.begin(), FitTemp.end());

		psize = psizeParam; //  strategy  C3

		int psizeval = NInds * psize;
		if (psizeval <= 1)
			psizeval = 2;

		for (int TheChosenOne = 0; TheChosenOne != NInds; TheChosenOne++) {
			MemoryCurrentIndex = IntRandom(MemorySize + 1);
			do
				prand = Indexes[IntRandom(psizeval)];
			while (prand == TheChosenOne && (double)NFEval / (double)MaxFEval < 0.5);

			int Rand1;
			int Rand2;
			do
				Rand1 = Indexes[ComponentSelector(generator_uni_i_2)];    //  strategy  H
			while (Rand1 == prand);
			do
				Rand2 = Indexes[ComponentSelector(generator_uni_i_2)];
			while (Rand2 == prand || Rand2 == Rand1);
			do {
				if (MemoryCurrentIndex < MemorySize)
					F = CachyRand(MemoryF[MemoryCurrentIndex], 0.1);
				else
					F = CachyRand(0.9, 0.1);   // strategy  A
			} while (F < 0.0);
			if (F > 1.0)
				F = 1.0;
			     if ((double)NFEval / (double)MaxFEval < 0.6 && F > 0.7)  // streagy  G
			         F = 0.7;

			F2 = 1.0 * F;   //


			if (Random(0, 1) < (double)CurrentArchiveSize / ((double)CurrentArchiveSize + (double)NInds)) {
				Rand2 = IntRandom(CurrentArchiveSize);
				for (int j = 0; j != NVars; j++) {
					Donor[j] = Popul[TheChosenOne][j] +
						F2 * (Popul[prand][j] - Popul[TheChosenOne][j]) +
						F * (Popul[Rand1][j] - Archive[Rand2][j]);
				}
			}
			else {
				for (int j = 0; j != NVars; j++) {
					Donor[j] = Popul[TheChosenOne][j] +
						F2 * (Popul[prand][j] - Popul[TheChosenOne][j]) +
						F * (Popul[Rand1][j] - Popul[Rand2][j]);
				}
			}
			FindLimits(Donor, Popul[TheChosenOne], NVars, Left, Right);

			int WillCrossover = IntRandom(NVars);
			if (MemoryCurrentIndex < MemorySize) {
				if (MemoryCr[MemoryCurrentIndex] < 0)
					Cr = 0;
				else
					Cr = NormRand(MemoryCr[MemoryCurrentIndex], 0.1);
			}
			else
				Cr = NormRand(0.9, 0.1);  // strategy  A
			if (Cr >= 1)
				Cr = 1;
			if (Cr <= 0)
				Cr = 0;

			      if ((double)NFEval / (double)MaxFEval < 0.25)  // strategy  G
			         Cr = max(Cr, 0.7);
			      if ((double)NFEval / (double)MaxFEval < 0.5)
			         Cr = max(Cr, 0.6);
				   //iSHADE_RSP
			bool perturbation = rand() / (double)RAND_MAX < jumping_rate;  // strategy  I
			for (int j = 0; j != NVars; j++) {
				if (Random(0, 1) < Cr || WillCrossover == j)
					PopulTemp[TheChosenOne][j] = Donor[j];
				else
					PopulTemp[TheChosenOne][j] = perturbation ? CachyRand(Popul[TheChosenOne][j], 0.1) : Popul[TheChosenOne][j];
			}
			FitTemp[TheChosenOne] = cec_17_(PopulTemp[TheChosenOne], func_num);
			if (FitTemp[TheChosenOne] <= globalbest)
				globalbest = FitTemp[TheChosenOne];
			if (FitTemp[TheChosenOne] <= FitMass[TheChosenOne])
				SaveSuccessCrF(Cr, F, fabs(FitMass[TheChosenOne] - FitTemp[TheChosenOne]));
			if (NFEval % 100 == 0) {
				double temp = globalbest - fopt[0];
				if (temp <= 10E-8)
					temp = 0;
				outFile << temp << ",";
			}
			/////////////////////////////mainmainmainmainmain/////////////////////////////
			/////////////////////////////mainmainmainmainmain/////////////////////////////
			/////////////////////////////mainmainmainmainmain/////////////////////////////
			double diffNorm1 = 0;
			double diffNorm2 = 0;
			int pBestRate = psizeval + 0.1*NInds;///////////////pBestRate 21%
			int pworst = Indexes[NInds-1];
			int  DisRate = 0.85*NInds;///////////////DisRate 15%
			if(NFEval < 0.3*MaxFEval){///////////////nFES 30%
				if (std::find(Indexes-1, Indexes + DisRate-1, TheChosenOne) != Indexes + DisRate-1) {//Individuals outside of distance differencing operations
					if (FitTemp[TheChosenOne] <= FitMass[TheChosenOne]) {
						CopyToArchive(Popul[TheChosenOne], FitMass[TheChosenOne]);
						for (int j = 0; j != NVars; j++)
							Popul[TheChosenOne][j] = PopulTemp[TheChosenOne][j];
						FitMass[TheChosenOne] = FitTemp[TheChosenOne];
					}
				}
				else{//Individuals of distance differencing operations
					if (TheChosenOne == pworst){//Global Best and Worst Individual Disturbance Operations
						for (int j = 0; j != NVars; j++){
								double diff1 = PopulTemp[TheChosenOne][j] - Popul[Indexes[0]][j];
								diffNorm1 += diff1 * diff1;
								double diff2 = Popul[TheChosenOne][j] - Popul[Indexes[0]][j];
								diffNorm2 += diff2 * diff2;
						}//CWorstIND
						if (diffNorm1 < diffNorm2){
							CopyToArchive(Popul[TheChosenOne], FitMass[TheChosenOne]);
							for (int j = 0; j != NVars; j++)
								Popul[TheChosenOne][j] = PopulTemp[TheChosenOne][j];							
							FitMass[TheChosenOne] = FitTemp[TheChosenOne];
						}
					}
					else{//Disturbance operations of some of the best and some of the worst individuals
						for (int j = 0; j != NVars; j++){
							for (int PBi = 0; PBi != pBestRate; PBi++){
								double diff1 = PopulTemp[TheChosenOne][j] - Popul[Indexes[PBi]][j];
								diffNorm1 += diff1 * diff1;
								double diff2 = Popul[TheChosenOne][j] - Popul[Indexes[PBi]][j];
								diffNorm2 += diff2 * diff2;
							}//PBestIND
						}
						if(diffNorm1 < diffNorm2){
							CopyToArchive(Popul[TheChosenOne], FitMass[TheChosenOne]);
							for (int j = 0; j != NVars; j++)
								Popul[TheChosenOne][j] = PopulTemp[TheChosenOne][j];							
							FitMass[TheChosenOne] = FitTemp[TheChosenOne];
						}
					}
					
					
				}//0.9popu
			}//0.3NFEVal
			else{
				if (FitTemp[TheChosenOne] <= FitMass[TheChosenOne]) {
					CopyToArchive(Popul[TheChosenOne], FitMass[TheChosenOne]);
					for (int j = 0; j != NVars; j++)
						Popul[TheChosenOne][j] = PopulTemp[TheChosenOne][j];
					FitMass[TheChosenOne] = FitTemp[TheChosenOne];
				}
			}
		}//TheChosenOne

		/*
		for (int TheChosenOne = 0; TheChosenOne != NInds; TheChosenOne++) {
			if (FitTemp[TheChosenOne] <= FitMass[TheChosenOne]) {
				CopyToArchive(Popul[TheChosenOne], FitMass[TheChosenOne]);
				for (int j = 0; j != NVars; j++)
					Popul[TheChosenOne][j] = PopulTemp[TheChosenOne][j];
				FitMass[TheChosenOne] = FitTemp[TheChosenOne];
			}
		}
		*/

		int newNInds = int(double(NIndsMin - NIndsMax) / MaxFEval * NFEval + NIndsMax);
		if (newNInds < NIndsMin)
			newNInds = NIndsMin;
		if (newNInds > NIndsMax)
			newNInds = NIndsMax;
		int newArchSize = double(MaxFEval - NFEval) / (double)MaxFEval * (ArchiveSizeParam * (NIndsMax - NIndsMin));
		if (newArchSize < NIndsMin)
			newArchSize = NIndsMin;
		ArchiveSize = newArchSize;
		if (CurrentArchiveSize >= ArchiveSize)
			CurrentArchiveSize = ArchiveSize;
		RemoveWorst(NInds, newNInds);
		NInds = newNInds;
		UpdateMemoryCrF();
		SuccessFilled = 0;
		Generation++;
	} while (NFEval < MaxFEval);
}
void Optimizer::Clean() {
	delete Donor;
	delete Trial;
	delete Rands;
	for (int i = 0; i != NIndsMax; i++) {
		delete Popul[i];
		delete PopulTemp[i];
	}
	for (int i = 0; i != NIndsMax * Int_ArchiveSizeParam; i++)
		delete Archive[i];
	delete Archive;
	delete Popul;
	delete PopulTemp;
	delete FitMass;
	delete FitMassTemp;
	delete FitMassCopy;
	delete BestInd;
	delete Indexes;
	delete tempSuccessCr;
	delete tempSuccessF;
	delete FitDelta;
	delete MemoryCr;
	delete MemoryF;
}
int main() {
	cout << "Random seeds are:" << endl;
	cout << seed1 << "\t" << seed2 << "\t" << seed3 << "\t" << seed4 << "\t" << seed5 << "\n";
	clock_t starttime,endtime;
	

	//for (int GNVarsIter = 0; GNVarsIter != 4; GNVarsIter++) {
	int GNVarsIter = 1; {

		if (GNVarsIter == 0)
			GNVars = 10;
		if (GNVarsIter == 1)
			GNVars = 30;
		if (GNVarsIter == 2)
			GNVars = 50;
		if (GNVarsIter == 3)
			GNVars = 100;
		MaxFEval = GNVars * 10000;
		cout << "Run D_" << GNVars << "\n";
		Optimizer OptZ;
		double* xopt = new double[GNVars];

		starttime = clock();
		for (int func_num = 1; func_num < 31; func_num++) {
			/*if (func_num == 2)
				continue;  //F2 is excluded from the competition!
			cout << "Function " << func_num << "\t"
				<< "D" << GNVars << "\n"
				<< "Runs: " << endl;
			sprintf(buffer, "LSHADE_RSP_%d_%d.txt", func_num, GNVars);*/
			stringstream ss;
			string pro;
			pro = to_string(func_num);
			ss << GNVars;
			string tmp(ss.str());
			string fileNameStr = "RSDE_D" + tmp + "p" + pro + ".csv";
			char fileName[500];
			strcpy(fileName, fileNameStr.c_str());
			cout << fileNameStr << "" << fileName;
			ofstream outFile;
			outFile.open(fileNameStr, ios::app);

			ofstream fout(buffer);
			for (int RunN = 0; RunN != 51; RunN++) {
				cout << RunN << "\t";
				GetOptimum(func_num, xopt, fopt);
				globalbestinit = false;
				initfinished = false;
				LastFEcount = 0;
				NFEval = 0;
				double NewPopSize = int(18 * GNVars);  // strategy  B1
				double NewArchSize = 1.0;
				double NewArchProb = 0.25;
				double NewPsize = 0.11;  // strategy  C3
				OptZ.Initialize(NewPopSize, GNVars, func_num, NewArchSize, NewArchProb, NewPsize);
				OptZ.MainCycle(outFile);
				OptZ.Clean();
			}
			cout << endl;
			/*for (int i = 0; i != 14; i++) {
				for (int j = 0; j != 51; j++)
					fout << ResultsArray[j][i] << "\t";
				fout << "\n";
			}*/
		}
		endtime = clock();
		ofstream fout_t("time.txt", ios::app);
		fout_t <<" D" << GNVars <<" Algorithm complexity is " << (double)(endtime - starttime) / CLOCKS_PER_SEC << "s" << endl;
		//fout_t << "fo_rate" << fo_rate <<" D" << GNVars <<" Algorithm complexity is " << (double)(endtime - starttime) / CLOCKS_PER_SEC << "s" << endl;
		cout << endl;
		delete xopt;
	}
	return 0;
}