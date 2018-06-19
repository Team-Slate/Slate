#include <opencv2/highgui.hpp>
#include <bits/stdc++.h>
#include <opencv2/imgproc.hpp>
#include <fstream>
#define ROW 720
#define COLUMN 1280
#define scaleX 30
#define scaleY 30
using namespace cv;
using namespace std;
String path="trainingImages/";
//String path="/home/anubhaw/strings/";
String pathStrings="strings/";
String pathShapes="shapes/";
int scalingFactor=10;
int boundary[4];//[0]minX, [1]maxX, [2]minY, [3]maxY
Mat original(ROW,COLUMN,CV_8UC3);
Mat red(ROW,COLUMN,CV_8UC3);
Mat canavas(ROW,COLUMN,CV_8UC1);
Mat resolved(ROW/scalingFactor,COLUMN/scalingFactor,CV_8UC1);
Mat scaled(scaleY,scaleX,CV_8UC1);
Mat filtered(scaleY,scaleX,CV_8UC1);
Mat readImage(scaleY,scaleX,CV_8UC1);
Mat calculator(ROW,COLUMN,CV_8UC1);
Mat heading(ROW,COLUMN,CV_8UC1);
Mat canavasBackup(ROW,COLUMN,CV_8UC1);
Mat lastCanavas(ROW,COLUMN,CV_8UC1);
Mat eraserbBackup(ROW,COLUMN,CV_8UC1);
vector <Mat> spl;
long long squaredDistance[100000];
int nImagesOfEachType=2;
//int setOfLabels[]={48,49,50,51,52,53,54,55,57,65,0};
int setOfLabels[]={'0','1','2','3','4','5','6','7','9','-','<','+','A','C','I','X','W','M','=',97,0};
int setOfLabelSlate[]={'0','1','2','3','4','5','6','7','9','-','<','+','A','C','I','X','W','M','=',97,0};
int setOfLabelsDigits[]={'0','1','2','3','4','5','6','7','9','<',0};
int setOfLabelsCalculatorSymbols[]={'-','*','=','0','<','X',0};
int setOfLabelsCalculatorTotal[]={'0','1','2','3','4','5','6','7','9','-','<','X','+','=',97,0};
int labelTotal[260];
int labelSlate[260];
int labelDigits[260];
int labelCalculatorSymbols[260];
int labelCalculatorTotal[260];
typedef struct node
{   
	char id;
	Mat imageData;
	int imageArray[scaleY][scaleX];
	int comX,comY;
	int nPoints;
	int nPointsAbove,nPointsBelow,nPointsLeft,nPointsRight;
}element;
element capturedImage;
vector <element> trainedData;
vector <element> ::iterator iteratorTrainedData[260];
void sampleFrameDraw(Mat);
void invertImage(Mat);
void initializeMatObject(Mat);
void thresholding(Mat);
void setResolution(Mat,Mat);
void findBoundary(Mat);
void scaling(Mat,Mat);
void filtering(Mat,Mat);
void initializeImageToStruct(element*,Mat);
void makeArrayFromMat(element*);
int findNearestPixel(vector <element>::iterator,int,int,int*,int*);
long long calculateSquaredDistance(vector <element>::iterator,element*);
int squaredDistaceLogic(int*,int*);
void displaySquaredDistace(int*);
int recognition(Mat,int*);
int writeImage(int,int);
void getImageForTraining(element*,int,int);
void setId(element*,int);
void setComnPoints(element*);
void setnPoints(element *);
void training();
void displayStructData(element*);
void displayTrainedData();
void setIteratorValuesFromVectorElement(vector <element>::iterator *,vector <element>::iterator,vector <element>::iterator);
void getImageIntoMat(String,Mat*);
void displayImageAtPosition(Mat*,Mat,int,int);
void displayHeading(Mat*,int,int);
void displayResultInCalculator(int,vector <Mat>,vector <element> :: iterator *,Mat*);
void calculatorActivity(vector <element> ::iterator*);
void musicActivity();
void eraser(Mat);
void displayImageAtPositionInImageBoard(Mat*,Mat,int,int);
void imageBoardActivity();
void asciiArtActivity();
int secondActivity(int);
void initializeArray(int*,int);
void setLabelFrequency(int*,int*);
VideoCapture cap(0);
void sampleFrameDraw(Mat resolved)
{
	cout<<"sampleFrameDraw"<<endl;
	int i,j,c=0;
	for(i=2;i<7;i++)
	{
		unsigned char *p=resolved.ptr(i);
		for(j=c;j<c+10;j+=2)
		{	
			*(p+j)=250;
		}
		c=!c;
	}
}
void invertImage(Mat original)
{
	//cout<<"invertImage()"<<endl;
	int i,j,temp;
	for(i=0;i<original.rows;i++)
	{
		unsigned char *p=original.ptr(i);
		for(j=0;j<original.cols/2;j++)
		{
			temp=*(p+(3*j));
			*(p+(3*j))=*(p+(3*(original.cols-j-1)));
			*(p+(3*(original.cols-j-1)))=temp;
			temp=*(p+(3*j)+1);
			*(p+(3*j)+1)=*(p+(3*(original.cols-j-1))+1);
			*(p+(3*(original.cols-j-1))+1)=temp;
			temp=*(p+(3*j)+2);
			*(p+(3*j)+2)=*(p+(3*(original.cols-j-1))+2);
			*(p+(3*(original.cols-j-1))+2)=temp;
		}
	}
}
void initializeMatObject(Mat obj)
{
	//cout<<"initializeMatObject()"<<endl;
	int i,j;
	for(i=0;i<obj.rows;i++)
	{
		unsigned char *p=obj.ptr(i);
		for(j=0;j<obj.cols;j++)
		{
			*(p+j)=0;
		}
	}
}
void thresholding(Mat canavas)
{
	//cout<<"thresholding()"<<endl;
	int i,j; 
	for(i=0;i<spl[2].rows;i++)
	{
		unsigned char *p=spl[2].ptr(i);
		unsigned char *g=canavas.ptr(i);
		for(j=0;j<spl[2].cols;j++)
		{
			if(*(p+j)>250)
			{
				//cout<<j<<" "<<i<<endl;
				*(g+j)=250;
			}
		}
	}
}
void setResolution(Mat canavas,Mat resolved)//decrease the size of image by scaling factor
{
	cout<<"setResolution()"<<endl;
	int row=0,col,i,j,k,l,s;
	unsigned char *q=resolved.ptr(0);
	for(i=0;i<ROW;i+=scalingFactor)
	{
		col=0;
		unsigned char *q=resolved.ptr(row);
		for(j=0;j<COLUMN;j+=scalingFactor)
		{
			s=0;
			for(k=i;k<i+scalingFactor;k++)
			{
				unsigned char *p=canavas.ptr(k);
				for(l=j;l<j+scalingFactor;l++)
				{
					s+=(*(p+l)/250);
				}
			}
			if(s>=(scalingFactor))
			{
				*(q+col)=250;
			}
			col++;
		}
		row++;	
	}
}
void findBoundary(Mat resolved)//find the boundar of image
{
	cout<<"findBoundary()"<<endl;
	int i,j;
	boundary[0]=2000;
	boundary[1]=-1;
	boundary[2]=2000;
	boundary[3]=-1;
	for(i=0;i<resolved.rows;i++)
	{
		unsigned char *p=resolved.ptr(i);
		for(j=0;j<resolved.cols;j++)
		{
			if(*(p+j)==250)
			{
				if(j<boundary[0])
				{
					boundary[0]=j;
				}
				if(j>boundary[1])
				{
					boundary[1]=j;
				}
				if(i<boundary[2])
				{
					boundary[2]=i;
				}
				if(i>boundary[3])
				{
					boundary[3]=i;
				}
			}
		}
	}
	for(i=0;i<4;i++)
	{
		cout<<boundary[i]<<" ";
	}
	cout<<endl;
}
void scaling(Mat resolved,Mat scaled)//convert the image to scaleX*scaleY
{
	cout<<"scaling()"<<endl;
	int a[resolved.rows][scaleX],b[scaleY][scaleX],col,row,s;
	for(int i=0;i<scaleY;i++)
	{
		for(int j=0;j<scaleX;j++)
		{
			b[i][j]=0;
		}
	}
	for(int i=0;i<resolved.rows;i++)
	{
		for(int j=0;j<scaleX;j++)
		{
			a[i][j]=0;
		}
	}
	//scaling columns
	// imshow("resolved",resolved);
	if(boundary[1]-boundary[0]>scaleX)
	{
		//cout<<"if(boundary[1]-boundary[0]>scaleX)"<<endl;
		float localScalingFactor=(boundary[1]-boundary[0]+1)/(float)scaleX;
		//cout<<"localScalingFactor="<<localScalingFactor<<endl;
		for(int i=0;i<resolved.rows;i++)
		{
			col=0;
			unsigned char *p=resolved.ptr(i);
			for(float j=boundary[0];j<boundary[1];j+=localScalingFactor)
			{
				s=0;
				for(int k=j;k<(int)(j+localScalingFactor);k++)
				{
					s+=(*(p+k)/250);
				}

				if(s>(localScalingFactor/2))
				{
					a[i][col]=250;
				}
				else
				{
					a[i][col]=0;
				}
				col++;
			}
		}	
	}
	else
	{
		for(int i=0;i<resolved.rows;i++)
		{
			int col=(scaleX-(boundary[1]-boundary[0]))/2;
			//cout<<"i="<<i<<"col="<<col<<" "<<resolved.rows<<endl;
			unsigned char *p=resolved.ptr(i);
			for(int j=boundary[0];j<boundary[1];j++)
			{
				//cout<<"j="<<j<<endl;
				if(*(p+j)==250)
				{
					a[i][col]=250;
				}
				else
				{
					a[i][col]=0;
				}
				col++;
			}
		}
	}
	//cout<<"Column scaling done\n";
	//scaling rows
	if(boundary[3]-boundary[2]>scaleY)
	{
	//	cout<<"if(boundary[2]-boundary[1]>scaleY)"<<endl;
		float localScalingFactor=(boundary[3]-boundary[2]+1)/(float)scaleY;
		//cout<<"localScalingFactor="<<localScalingFactor<<endl;
		for(int i=0;i<scaleX;i++)
		{
			row=0;
			for(float j=boundary[2];j<boundary[3];j+=localScalingFactor)
			{	s=0;
				for(int k=j;k<j+localScalingFactor;k++)
				{
					s+=(a[k][i]/250);
				}
				if(s>(localScalingFactor/2))
				{
					b[row][i]=250;
				}
				else
				{
					b[row][i]=0;
				}
				row++;
			}

		}
	}
	else
	{
		
		for(int i=0;i<scaleX;i++)
		{
			int row=(scaleY-(boundary[3]-boundary[2]))/2;
			//cout<<"row="<<row<<endl;
			for(int j=boundary[2];j<boundary[3];j++)
			{
				if(a[j][i]==250)
				{
					b[row][i]=250;
				}
				else
				{
					b[row][i]=0;
				}
				row++;
			}
		}
	}
	//cout<<"Column scaling done\n";
	for(int i=0;i<scaleY;i++)
	{
		unsigned char *p=scaled.ptr(i);
		for(int j=0;j<scaleX;j++)
		{
			/*if(b[i][j]==250)
			{
				cout<<"1 ";
			}
			else
			{
				cout<<"0 ";
			}*/
			*(p+j)=(int)b[i][j];
		}
		//cout<<endl;
	}
	return;
}
void filtering(Mat scaled,Mat filtered)
{
	cout<<"filtering()"<<endl;
	for(int i=0;i<scaled.rows;i++)
	{
		unsigned char *p=scaled.ptr(i);
		unsigned char *r=filtered.ptr(i);
		for(int j=0;j<scaled.cols;j++)
		{
			*(r+j)=*(p+j);
		}
	}
	for(int i=1;i<scaled.rows-1;i++)
	{
		unsigned char *o=scaled.ptr(i-1);
		unsigned char *p=scaled.ptr(i);
		unsigned char *q=scaled.ptr(i+1);
		unsigned char *r=filtered.ptr(i);
		for(int j=1;j<scaled.cols-1;j++)
		{
			int s=*(p+j+1)/250+*(p+j-1)/250+*(o+j-1)/250+*(o+j)/250+*(o+j+1)/250+*(q+j-1)/250+*(q+j)/250+*(q+j+1)/250;
			//cout<<"s="<<s<<endl;
			if(s==0)
			{
				*(r+j)=0;
			}

		}
	}
}
void initializeImageToStruct(element *temp,Mat filtered)
{
	cout<<"initializeImageToStruct()"<<endl;
	temp->imageData=filtered;
}
void makeArrayFromMat(element * temp)
{
	cout<<"makeArrayFromMat()"<<endl;
	for(int i=0;i<scaleY;i++)
	{
		unsigned char *p=temp->imageData.ptr(i);
		for(int j=0;j<scaleX;j++)
		{
			temp->imageArray[i][j]=*(p+j);
			//cout<<temp->imageArray[i][j]<<" ";
		} 
		//cout<<endl;
	}
}
int findNearestPixel(vector <element> ::iterator it,int i,int j,int *y,int *x)
{
	//cout<<"findNearestPixel("<<i<<","<<j<<") "<<"it->id = "<<it->id<<endl;
	pair < int,int > p;
	queue <pair < int,int > > qe;
	int tempX,tempY;
	int bfsArray[scaleY][scaleX];
	for(int i=0;i<scaleY;i++)
	{
		for(int j=0;j<scaleX;j++)
		{
			bfsArray[i][j]=0;
		}
	}
	p.first=i;
	p.second=j;
	qe.push(p);
	while(!qe.empty())
	{
		tempY=(qe.front()).first;
		tempX=(qe.front()).second;
		if(it->imageArray[tempY][tempX]==250)
		{
			*y=tempY;
			*x=tempX;
			return 1;
		}
		qe.pop();
		bfsArray[tempY][tempX]=1;
		if(tempX!=0 && bfsArray[tempY][tempX-1]==0)
		{
			p.second=tempX-1;
			p.first=tempY;
			qe.push(p);
		}
		if(tempY!=0 && bfsArray[tempY-1][tempX]==0)
		{
			p.second=tempX;
			p.first=tempY-1;
			qe.push(p);
		}
		if(tempX!=scaleX-1 && bfsArray[tempY][tempX+1]==0)
		{
			p.second=tempX+1;
			p.first=tempY;
			qe.push(p);
		}
		if(tempY!=scaleY-1 && bfsArray[tempY+1][tempX]==0)
		{
			p.second=tempX;
			p.first=tempY+1;
			qe.push(p);
		}
	}
	return 0;
}
long long calculateSquaredDistance(vector <element> ::iterator it,element* temp)
{
	cout<<"calculateSquaredDistance("<<it->id<<")"<<endl;
	long long distance=0;
	int x=0,y=0;
	for(int i=0;i<scaleY;i++)
	{
		for(int j=0;j<scaleX;j++)
		{
			
			if(temp->imageArray[i][j])
			{
				if(findNearestPixel(it,i,j,&y,&x))
				{
					//cout<<i<<" "<<j<<" "<<y<<" "<<x<<" "<<(abs(i-y)*abs(i-y))+(abs(j-x)*abs(j-x))<<endl;
					distance+=(abs(i-y)*abs(i-y))+(abs(j-x)*abs(j-x));
				}
			}
		}
	}
	return distance;
}
int squaredDistaceLogic(int *squaredDistanceLocal,int *labelToAccept)
{
	cout<<"squaredDistaceLogic()"<<endl;
	int count=0;
	int min=20000;
	vector <element> :: iterator minit;
	for(vector <element> ::iterator it=trainedData.begin();it!=trainedData.end();it++)
	{
		if(*(labelToAccept+(it->id))==0)
		{
			continue;
		}
		cout<<it->id<<" ";
		squaredDistance[count]=calculateSquaredDistance(it,&capturedImage);
		if(squaredDistance[count]<min)
		{
			min=squaredDistance[count];
			minit=it;
		}
		count++;
	}
	cout<<endl;
	*squaredDistanceLocal=min;
	return minit->id;
}
void displaySquaredDistace(int *labelToAccept)
{
	cout<<"displaySquaredDistace()"<<endl;
	int count=0;
	for(vector <element> ::iterator it=trainedData.begin();it!=trainedData.end();it++)
	{
		if(*(labelToAccept+(it->id))==0)
		{
			continue;
		}
		cout<<"id = "<<it->id<<" squaredDistance = "<<squaredDistance[count]<<endl;
		count++;
	}
}
int recognition(Mat canavas,int *labelToAccept)
{
	cout<<"recognition()"<<endl;
	int squaredDistanceLocal;
	setResolution(canavas,resolved);
		namedWindow("resolved",WINDOW_NORMAL);
		 imshow("resolved",resolved);
	findBoundary(resolved);
		cout<<boundary[0]<<" "<<boundary[1]<<" "<<boundary[2]<<" "<<boundary[3]<<endl;
		cout<<boundary[1]-boundary[0]<<" "<<boundary[3]-boundary[2]<<endl;
	scaling(resolved,scaled);
		 namedWindow("scaled",WINDOW_NORMAL);	
		 imshow("scaled",scaled);
	filtering(scaled,filtered);
		 namedWindow("filtered",WINDOW_NORMAL);	
		 imshow("filtered",filtered);

	initializeImageToStruct(&capturedImage,filtered);
	setComnPoints(&capturedImage);
	setnPoints(&capturedImage);
	makeArrayFromMat(&capturedImage);
	displayStructData(&capturedImage);

	int recognisedChar=squaredDistaceLogic(&squaredDistanceLocal,labelToAccept);
	displaySquaredDistace(labelToAccept);
	if(squaredDistanceLocal<1000)
	{
		return recognisedChar;
	}
	else 
		if((capturedImage.nPoints)>230)
		{
			return 56;
		}
		else
		{
			return 0;
		}
	//cout<<"displayed data"<<endl;
}
int writeImage(int ch,int count)
{
	
	cout<<"writeImage()"<<endl;
	initializeMatObject(resolved);
	stringstream sa,sb;
	sa<<count;
	sb<<ch;
	string str1=sa.str();
	string str2=sb.str();
	setResolution(canavasBackup,resolved);
	findBoundary(resolved);
	scaling(resolved,scaled);
	namedWindow("scaled",WINDOW_NORMAL);
	//imshow("scaled",scaled);
	//bool isSuccess = imwrite(path+str2+"_"+str1+".jpg",scaled);
	bool isSuccess = imwrite(pathStrings+"down.jpg",scaled);

	if (isSuccess == false)
	{
  		cout << "Failed to save the image" << endl;
  		return -1;
 	}
 	return 1;
}
void getImageForTraining(element *temp,int id,int count)
{
	cout<<"getImageForTraining("<<id<<","<<count<<")"<<endl;
	stringstream sa,sb;
	sa<<count;
	sb<<id;
	String str1=sa.str();
	String str2=sb.str();
	String localPath=path+str2+"_"+str1+".jpg";
	cout<<localPath<<endl;
	temp->imageData = imread(localPath,COLOR_BGR2GRAY);
	if((temp->imageData).empty())
	{
		cout<<"Failed to open image"<<endl;
	}
	
	//cout<<temp->imageData;
	//while(1)
	//{}
	//cout<<"4\n";
}
void setId(element *temp,int id)
{
	cout<<"setId("<<id<<")"<<endl;
	temp->id=id;
}
void setComnPoints(element *temp)
{
	cout<<"setComnPoints()"<<endl;
	int sx=scaleX/2,sy=scaleY/2;
	int count=0;
	for(int i=0;i<scaleY;i++)
	{
		unsigned char *p=(temp->imageData).ptr(i);
		//cout<<temp->imageData;
		for(int j=0;j<scaleX;j++)
		{
			if(*(p+j)>=220)
			{
				*(p+j)=250;
				sx+=(j-(scaleX/2));
				sy+=(i-(scaleY/2));
				//cout<<sx<<" "<<sy<<" "<<count<<endl;
				count++;
			}
			else
			{
				*(p+j)=0;
			}
		}	
	}
	
	temp->comX=sx/count;
	temp->comY=sy/count;
	temp->nPoints=count;
	//cout<<"2\n";
}
void setnPoints(element *temp)
{
	cout<<"setnPoints()"<<endl;
	int quad1=0,quad2=0,quad3=0,quad4=0;
	for(int i=0;i<scaleY/2;i++)
	{
		unsigned char *p=(temp->imageData).ptr(i);
		for(int j=0;j<scaleX/2;j++)
		{
			if(*(p+j)==250)
			{
				quad1++;
			}
		}	
	}
	for(int i=0;i<scaleY/2;i++)
	{
		unsigned char *p=(temp->imageData).ptr(i);
		for(int j=scaleX/2;j<scaleX;j++)
		{
			if(*(p+j)==250)
			{
				quad2++;
			}
		}	
	}
	for(int i=scaleY/2;i<scaleY;i++)
	{
		unsigned char *p=(temp->imageData).ptr(i);
		for(int j=0;j<scaleX/2;j++)
		{
			if(*(p+j)==250)
			{
				quad3++;
			}
		}	
	}
	for(int i=scaleY/2;i<scaleY;i++)
	{
		unsigned char *p=(temp->imageData).ptr(i);
		for(int j=scaleX/2;j<scaleX;j++)
		{
			if(*(p+j)==250)
			{
				quad4++;
			}
		}	
	}
	temp->nPointsRight=quad2+quad4;
	temp->nPointsLeft=quad1+quad3;
	temp->nPointsAbove=quad1+quad2;
	temp->nPointsBelow=quad3+quad4;
}
void training()
{
	cout<<"training()"<<endl;
	for(int id=0;setOfLabels[id]!=0;id++)
	{
		for(int count=0;count<nImagesOfEachType;count++)
		{
			cout<<"id="<<setOfLabels[id]<<" count="<<count<<endl;
			element temp;
			setId(&temp,setOfLabels[id]);
			getImageForTraining(&temp,setOfLabels[id],count);
			//cout<<"1"<<endl;
			setComnPoints(&temp);
			//cout<<temp.imageData;
			//namedWindow("orr",WINDOW_NORMAL);
			//imshow("orr",temp.imageData);
			//while(1)
			//{}
			setnPoints(&temp);
			makeArrayFromMat(&temp);
			trainedData.push_back(temp);
			//cout<<"done"<<endl;
		}
	}
	int id=56;
	int count=0;
	cout<<"id=8"<< "count=0"<<endl;
	element temp;
	setId(&temp,56);		
	getImageForTraining(&temp,id,count);
	//cout<<"1"<<endl;
	setComnPoints(&temp);
	//cout<<temp.imageData;
	//namedWindow("orr",WINDOW_NORMAL);
	//imshow("orr",temp.imageData);
	//while(1)
	//{}
	setnPoints(&temp);
	makeArrayFromMat(&temp);
	cout<<temp.imageData;
	trainedData.push_back(temp);
	cout<<"done"<<endl;
	id=56;
	count=1;
	cout<<"id=8"<< "count=1"<<endl;
	setId(&temp,56);		
	getImageForTraining(&temp,id,count);
	//cout<<"1"<<endl;
	setComnPoints(&temp);
	//cout<<temp.imageData;
	//namedWindow("orr",WINDOW_NORMAL);
	//imshow("orr",temp.imageData);
	//while(1)
	//{}
	setnPoints(&temp);
	makeArrayFromMat(&temp);
	cout<<temp.imageData;
	trainedData.push_back(temp);
	cout<<"done"<<endl;
	// id=56;
	// count=2;
	// cout<<"id=8"<< "count=2"<<endl;
	// setId(&temp,56);		
	// getImageForTraining(&temp,id,count);
	// //cout<<"1"<<endl;
	// setComnPoints(&temp);
	// //cout<<temp.imageData;
	// //namedWindow("orr",WINDOW_NORMAL);
	// //imshow("orr",temp.imageData);
	// //while(1)
	// //{}
	// setnPoints(&temp);
	// makeArrayFromMat(&temp);
	// cout<<temp.imageData;
	// trainedData.push_back(temp);
	// cout<<"done"<<endl;
	// id=56;
	// count=3;
	// cout<<"id=8"<< "count=3"<<endl;
	// setId(&temp,56);		
	// getImageForTraining(&temp,id,count);
	// //cout<<"1"<<endl;
	// setComnPoints(&temp);
	// //cout<<temp.imageData;
	// //namedWindow("orr",WINDOW_NORMAL);
	// //imshow("orr",temp.imageData);
	// //while(1)
	// //{}
	// setnPoints(&temp);
	// makeArrayFromMat(&temp);
	// cout<<temp.imageData;
	// trainedData.push_back(temp);
	// cout<<"done"<<endl;
}
void displayStructData(element* i)
{
	cout<<"displayStructData()"<<endl;
	//std::ofstream ofs;
	//ofs.open("captured.txt", std::ofstream::out | std::ofstream::trunc);
	//ofss<<"id = "<<i->id<<endl;
	cout<<"comX = "<<i->comX<<" comY = "<<i->comY<<endl;
	cout<<"nPoints = "<<i->nPoints<<endl;
	cout<<"nPointsAbove = "<<i->nPointsAbove<<endl;
	cout<<"nPointsBelow = "<<i->nPointsBelow<<endl;
	cout<<"nPointsLeft = "<<i->nPointsLeft<<endl;
	cout<<"nPointsRight = "<<i->nPointsRight<<endl;
	cout<<"nPointsAbove/nPointsBelow = "<<i->nPointsAbove/(float)i->nPointsBelow<<endl;
	cout<<"nPointsLeft/nPointsRight = "<<i->nPointsLeft/(float)i->nPointsRight<<endl;
	/*for(int k=0;k<scaleY;k++)
	{
		for(int j=0;j<scaleX;j++)
		{
			int temp=i->imageArray[k][j];
			if(temp)
			{
				cout<<temp<<" ";
			}
			else
			{
				cout<<temp<<"   ";
			}
			
		}
		cout<<endl;
	}
	cout<<"\n\n";*/
	//cout<<"done"<<endl;
	//ofs.close();
}
void displayTrainedData()
{
	cout<<"displayTrainedData()"<<endl;
	std::ofstream ofss;
		ofss.open("trainedData.txt", std::ofstream::out | std::ofstream::trunc);
	for(vector <element> ::iterator i=trainedData.begin();i!=trainedData.end();i++)
	{
		
		ofss<<"id = "<<i->id<<endl;
		ofss<<"comX = "<<i->comX<<" comY = "<<i->comY<<endl;
		ofss<<"nPoints = "<<i->nPoints<<endl;
		ofss<<"nPointsAbove = "<<i->nPointsAbove<<endl;
		ofss<<"nPointsBelow = "<<i->nPointsBelow<<endl;
		ofss<<"nPointsLeft = "<<i->nPointsLeft<<endl;
		ofss<<"nPointsRight = "<<i->nPointsRight<<endl;
		ofss<<"nPointsAbove/nPointsBelow = "<<i->nPointsAbove/(float)i->nPointsBelow<<endl;
		ofss<<"nPointsLeft/nPointsRight = "<<i->nPointsLeft/(float)i->nPointsRight<<endl;
		for(int k=0;k<scaleY;k++)
		{
			for(int j=0;j<scaleX;j++)
			{
				int temp=i->imageArray[k][j];
				if(temp)
				{
					ofss<<temp<<" ";
				}
				else
				{
					ofss<<temp<<"   ";
				}
				
			}
			ofss<<endl;
		}
		ofss<<"\n\n";
		ofss<<"done"<<endl;
	}
	ofss.close();
}
void setIteratorValuesFromVectorElement(vector <element> ::iterator *iteratorArray,vector <element> ::iterator begin,vector <element> ::iterator end)
{
	cout<<"setIteratorValuesFromVectorElement()"<<endl;	
	for(vector <element> ::iterator it=begin;it!=end;it+=nImagesOfEachType)
	{
		cout<<it->id<<" ";
		*(iteratorArray+(it->id))=it;
	}
	cout<<endl;
}
void getImageIntoMat(String path,Mat *heading)
{
	//cout<<"getImageIntoMat()"<<endl;
	*heading=imread(path,COLOR_BGR2GRAY);
	if((*heading).empty())
	{
		cout<<"Failed to open image "<<path<<endl;
	}
}
void displayImageAtPosition(Mat *canavas,Mat heading,int y,int x)
{
	//cout<<"displayImageAtPosition()"<<endl;
	for(int i=0;i<heading.rows;i++)
	{
		unsigned char *p=heading.ptr(i);
		unsigned char *q=(*canavas).ptr(i+y);
		for(int j=0;j<heading.cols;j++)
		{
			if(*(p+j)>220)
			{
				*(q+x+j)=250;
			}
		}
	}
}
void displayHeading(Mat *canavas,int y,int x)
{
	//cout<<"displayHeadingCalculator()"<<endl;
	
	displayImageAtPosition(canavas,heading,y,x);
}
void displayResultInCalculator(int result,vector <Mat> calculatorData,vector <element> ::iterator *iteratorTrainedData,Mat *canavas)
{
	int n=result,d=1,showX=10,num[100],x=0;
	if(result<0)
	{
		result=result*(-1);
		calculatorData.push_back(((*(iteratorTrainedData+'-'))->imageData).clone());
		cout<<"-done"<<endl;
	}
	n=result;
	while(n!=0)
	{
		d=n%10;
		n=n/10;
		num[x]=d;
		x++;
	}
	cout<<"1"<<endl;
	
	for(int i=x-1;i>=0;i--)
	{
		calculatorData.push_back(((*(iteratorTrainedData+num[i]+48))->imageData).clone());
	}
	cout<<"2"<<endl;
	for(vector <Mat> ::iterator it=calculatorData.begin();it!=calculatorData.end();it++)
	{
	 	displayImageAtPosition(canavas,*it,100,showX);
	 	showX+=40;
	}	

}
void calculatorActivity(vector <element> ::iterator *iteratorTrainedData)
{
	cout<<"calculatorActivity()"<<endl;

	Mat calculatorBackup(ROW,COLUMN,CV_8UC1);
	initializeMatObject(calculatorBackup);
	vector <Mat> calculatorData;
	int showX;
	getImageIntoMat(pathStrings+"calculator.jpg",&heading);
	//imshow("heading",heading);
	//namedWindow("heading",WINDOW_NORMAL);
	//namedWindow("canavasBackup",WINDOW_NORMAL);

	int recognisedChar;
	initializeMatObject(canavas);
	imshow("canavas",canavas);
	initializeMatObject(canavasBackup);
	//displayHeading(&calculatorBackup,y,x);
	int keyCount=0;
	int key[100];
	initializeArray(key,99);
	int digitCount=0;
	initializeMatObject(lastCanavas);
	while(1)
	{	
		//displayHeadingCalculator(canavas);
		//imshow("canavas",canavas);
	 	bool bSuccess=cap.read(original);
	 	if(bSuccess==false)
	 	{
	 	 	break;
	 	}
	 	initializeMatObject(canavas);
	 	canavas=canavasBackup.clone();
	 	invertImage(original);
	 	split(original,spl);
	 	thresholding(canavas);
	 	canavasBackup=canavas.clone();
	 	displayHeading(&canavas,5,128);
	 	showX=10;
	 	for(vector <Mat> ::iterator it=calculatorData.begin();it!=calculatorData.end();it++)
	 	{
	 		displayImageAtPosition(&canavas,*it,100,showX);
	 		showX+=40;
	 	}
	 	//imshow("original",original);
	 	//canavas=calculatorBackup.clone();
	 	imshow("red",spl[2]);
	 	imshow("canavas",canavas);
	 	//imshow("canavasBackup",canavasBackup);
	 	// while(waitKey(10)!=13)
	 	// {

	 	// }

	 	
	 	int keyPress=waitKey(10);
	 	if(keyPress==27)//Esc key
	 	{
	 		cout<<"clear"<<endl;
	 	 	initializeMatObject(canavas);
	 	 	initializeMatObject(canavasBackup);
	 	 	//system("clear");
	 	}
	  	if(keyPress==13)//Enter key
	 	{
	 	 	initializeMatObject(resolved);
	 	 	initializeMatObject(lastCanavas);
		 	
	 	 	recognisedChar=recognition(canavasBackup,labelCalculatorTotal) ;
	 	 	initializeMatObject(canavasBackup);
	 	 	cout<<"Recognised Character = "<<(char)recognisedChar<<endl;
	 	 	if(recognisedChar)
		 	{
		 		
		 		// if(recognisedChar=='8')
		 		// {
		 		// 	calculatorData.push_back(filtered.clone());
		 		// }
		 		// else
		 		
		 			calculatorData.push_back(((*(iteratorTrainedData+recognisedChar))->imageData).clone());
		 	
		 		if(recognisedChar>=48 && recognisedChar<=57)
		 		{
		 			key[keyCount]=(key[keyCount]*10)+(recognisedChar-48);
		 			cout<<"key["<<keyCount<<"]= "<<key[keyCount]<<endl;
		 			digitCount++;
		 		}
		 		else 
		 			if(recognisedChar=='+' || recognisedChar=='-' || recognisedChar=='/' || recognisedChar=='X')
		 			{
		 				keyCount++;
		 				key[keyCount]=recognisedChar;
		 				keyCount++;
		 				digitCount=0;
		 				
		 				
		 			}
		 			else 
		 				if(recognisedChar=='<')
		 				{
		 					initializeMatObject(canavas);
		 					return;
		 				}
		 				else 
		 					if(recognisedChar=='=')
		 					{
		 						break;
		 					}
		 					else
		 						if(recognisedChar==97)
		 						{
		 							if(digitCount==0)
		 							{
		 								digitCount=1;
		 								keyCount-=2;
		 							}
		 							else
		 							{
		 								key[keyCount]=0;
		 							}
		 							for(int i=0;i<digitCount;i++)
		 							{
		 								calculatorData.pop_back();
		 							}
		 							digitCount=0;
		 						}
		 	}
		 	else
		 	{
		 		cout<<"Unable to recognise"<<endl;
		 		initializeMatObject(canavasBackup);
		 	}
		} 
	}
	long long result=key[0];
	for(int i=1;i<keyCount;i+=2)
	{
		switch(key[i])
		{
			case '+':
				result=result+(key[i+1]);
				break;
			case '-':
				result=result-(key[i+1]);
				break;
			case '/':
				result=result/(key[i+1]);
				break;
			case 'X':
				result=result*(key[i+1]);
				break;			
		}
	}
	cout<<"Result = "<<result<<endl;
	initializeMatObject(canavas);
	displayHeading(&canavas,5,128);
	displayResultInCalculator(result,calculatorData,iteratorTrainedData,&canavas);
	imshow("canavas",canavas);
	waitKey(5000);
	initializeMatObject(canavas);

	return;
}
void musicActivity()
{
	system("canberra-gtk-play -f harryPotter.wav");
}
void eraser(Mat canavas)
{
	int i,j; 
	for(i=0;i<eraserbBackup.rows;i++)
	{
		unsigned char *g=canavas.ptr(i);
		unsigned char *h=eraserbBackup.ptr(i);
		
		for(j=0;j<eraserbBackup.cols;j++)
		{
			if(*(h+j)>=250)
			{
				//cout<<j<<" "<<i<<endl;

				*(g+j)=0;
			}
		}
	}
	initializeMatObject(eraserbBackup);
	for(i=0;i<spl[2].rows;i++)
	{
		unsigned char *p=spl[2].ptr(i);
		unsigned char *g=canavas.ptr(i);
		unsigned char *h=eraserbBackup.ptr(i);

		for(j=0;j<spl[2].cols;j++)
		{
			if(*(p+j)>250)
			{
				//cout<<j<<" "<<i<<endl;

				*(g+j)=250;
				*(h+j)=250;
			}
		}
	}
}
void displayImageAtPositionInImageBoard(Mat *canavas,Mat heading,int y,int x)
{
	//cout<<"displayImageAtPosition()"<<endl;
	for(int i=0;i<heading.rows;i++)
	{
		unsigned char *p=heading.ptr(i);
		unsigned char *q=(*canavas).ptr(i+y);
		for(int j=0;j<heading.cols;j++)
		{
			if(*(p+j)>220)
			{
				*(q+x+j)=250;
			}
			else
			{
				*(q+x+j)=0;
			}
		}
	}
}
void imageBoardActivity()
{
	Mat penMode;
	int currentPenMode='p';
	cout<<"imageBoardActivity()"<<endl;
	initializeMatObject(canavas);
	imshow("canavas",canavas);
	initializeMatObject(canavasBackup);
	int keyPress;
	while(1)
	{	
		keyPress=waitKey(10);
		if(keyPress=='p')
		{
			currentPenMode='p';
		}
		else
			if(keyPress=='e')
			{
				initializeMatObject(eraserbBackup);
				currentPenMode='e';
			}
	 	bool bSuccess=cap.read(original);
	 	if(bSuccess==false)
	 	{
	 	 	break;
	 	}
	 	invertImage(original);
	 	split(original,spl);
	 	if(currentPenMode=='p')
	 	{
	 		thresholding(canavas);
	 		getImageIntoMat(pathShapes+"pencil.jpg",&penMode);
	 	}
	 	else
	 	{
	 		eraser(canavas);
	 		getImageIntoMat(pathShapes+"eraser.jpg",&penMode);

	 	}
	 	displayImageAtPositionInImageBoard(&canavas,penMode,20,1100);
	 	imshow("red",spl[2]);
	 	imshow("canavas",canavas);
	 	if(keyPress==27)//Esc key
	 	{
	 		cout<<"clear"<<endl;
	 	 	initializeMatObject(canavas);
	 	}
	  	if(keyPress==8)//backspace Key
	 	{
	 	 	initializeMatObject(canavas);
	 	 	return;
		} 
	}
}
void asciiArtActivity()
{
	int WIDTH=180,HEIGHT=72,row,col;
	String as="  ....'''````,,,,^^^^:-_++++<>i!lI?/()1{}[]rcvunxzjftLCJUYXZO0Qoahkbdpqwm*WMB8&$#@";
	int aa[HEIGHT][WIDTH];
	while(1)
	{
		bool bSuccess=cap.read(original);
		if(bSuccess==false)
		{
		 	break;
		}
		invertImage(original);
		split(original,spl);
		//imshow("original",original);
		imshow("red",spl[2]);
		spl[2]=spl[0]/3+spl[1]/3+spl[2]/3;
		int keyPress=waitKey(10);
		if(keyPress==8)//Backspace key
		{
			system("clear");
			return;
		}
		row=0;
		col=0;
		for(int i=0;i<spl[2].rows;i+=((spl[2].rows)/HEIGHT))
		{
			col=0;
			for(int l=0;l<spl[2].cols;l+=((spl[2].cols)/WIDTH))
			{
				long long temp=0;
				for(int j=i;j<i+(spl[2].rows/HEIGHT);j++)
				{
					unsigned char * p=spl[2].ptr(j);
					for(int k=l;k<l+((spl[2].cols)/WIDTH);k++)
					{
						temp+=(*(p+k));
					}
					//cout<<aa[row][col]<<" ";
				}
				aa[row][col]=temp;
				aa[row][col]/=((spl[2].rows/HEIGHT)*(spl[2].cols/WIDTH));
				col++;
			}	
			row++;

		}
		int len=(255/as.length())+1;//4
		system("clear");
		for(int i=0;i<HEIGHT;i++)
		{
			for(int j=0;j<WIDTH;j++)
			{
				cout<<as[aa[i][j]/len];
			}
			cout<<endl;
		}
	}
}
int checkMatForNull()
{
	for(int i=0;i<spl[2].rows;i++)
	{
		unsigned char *p=spl[2].ptr(i);
		for(int j=0;j<spl[2].rows;j++)
		{
			if(*(p+j)==250)
			{
				return 1;
			}
		}
	}
	return 0;
}
void findCurrentPixelGlow(int *glowX,int *glowY)
{
	for(int i=0;i<spl[2].rows;i++)
	{
		unsigned char *p=spl[2].ptr(i);
		for(int j=0;j<spl[2].rows;j++)
		{
			if(*(p+j)>=230)
			{
				*glowX=j;
				*glowY=i;
				return;
			}
		}
	}
	*glowX=-1;
	*glowY=-1;
	
}
void brightnessActivity()
{
	cout<<"brightnessActivity()"<<endl;

	Mat brightnessBackup(ROW,COLUMN,CV_8UC1);
	Mat swipe(ROW,COLUMN,CV_8UC1);
	Mat to(ROW,COLUMN,CV_8UC1);
	Mat left(ROW,COLUMN,CV_8UC1);
	Mat right(ROW,COLUMN,CV_8UC1);
	Mat up(ROW,COLUMN,CV_8UC1);
	Mat down(ROW,COLUMN,CV_8UC1);
	getImageIntoMat(pathStrings+"swipe.jpg",&swipe);
	getImageIntoMat(pathStrings+"to.jpg",&to);
	getImageIntoMat(pathStrings+"left.jpg",&left);
	getImageIntoMat(pathStrings+"right.jpg",&right);
	getImageIntoMat(pathStrings+"up.jpg",&up);
	getImageIntoMat(pathStrings+"down.jpg",&down);
	//imshow("heading",heading);
	//namedWindow("canavasBackup",WINDOW_NORMAL);
	initializeMatObject(canavas);
	imshow("canavas",canavas);
	initializeMatObject(canavasBackup);
	//displayHeading(&calculatorBackup,y,x);
	int flag=0;
	int bFlag=0;
	int glowX,glowY;
	int startX,startY,endX,endY,xDiff,yDiff,lastY,lastX;
	int count=0;
	while(1)
	{	
		//displayHeadingCalculator(canavas);
		//imshow("canavas",canavas);
	 	bool bSuccess=cap.read(original);
	 	if(bSuccess==false)
	 	{
	 	 	break;
	 	}
	 	initializeMatObject(canavas);
	 	canavas=canavasBackup.clone();
	 	invertImage(original);
	 	split(original,spl);
	 	findCurrentPixelGlow(&glowX,&glowY);
	 	thresholding(canavas);
	 	canavasBackup=canavas.clone();
	 	//displayHeading(&canavas,5,128);
	 	displayImageAtPosition(&canavas,swipe,5,10);
	 	displayImageAtPosition(&canavas,swipe,50,10);
	 	displayImageAtPosition(&canavas,right,5,100);
	 	displayImageAtPosition(&canavas,left,50,100);
	 	displayImageAtPosition(&canavas,to,5,180);
	 	displayImageAtPosition(&canavas,to,50,180);
	 	displayImageAtPosition(&canavas,up,5,230);
	 	displayImageAtPosition(&canavas,down,50,230);
	 	//imshow("original",original);
	 	//canavas=calculatorBackup.clone();
	 	imshow("red",spl[2]);
	 	imshow("canavas",canavas);
	 //	imshow("canavasBackup",canavasBackup);
	 	// while(waitKey(10)!=13)
	 	// {

	 	// }

	 	int matStatus=checkMatForNull();
	 	if(!matStatus)
	 	{	
	 		if(count>=20)
	 		{
	 			if(flag==1)
	 			{
	 				if(endX- startX >0)
	 				{
	 					system("xbacklight -inc 20");		
	 				}
	 				else
	 				{
	 					system("xbacklight -dec 20");			
	 				}
	 				
	 			}
	 			flag=0;
	 			bFlag=0;
	 			initializeMatObject(canavasBackup);
	 			count=0;
	 		}
	 		else
	 		{
	 			count++;
	 		}
	 	}
	 	else
	 	{
	 		count=0;
	 		if(flag==0)
	 		{
	 			startX=glowX;
	 			startY=glowY;
	 			endX=glowX;
	 			endY=glowY;
	 		}
	 		else
	 		{
	 			endX=glowX;
	 			endY=glowY;
	 		}
	 		flag=1;
	 	}
		int keyPress=waitKey(10);
	  	if(keyPress==8)//Backspace key
	 	{
	 	 	return;
		} 
	}

	
}
int secondActivity(int recognisedChar)
{
	switch(recognisedChar)
	{
		case 'A':
			asciiArtActivity();
			return 1;
		case 'C':
			calculatorActivity(iteratorTrainedData);
			return 1;
		case 'I':
			imageBoardActivity();
			return 1;
		case 'M':
			musicActivity();
			return 1;
		case 'W':
			brightnessActivity();
			return 1;
		
	}
	return 0;
}
void initializeArray(int *labelToSet,int maxIndex)
{
	for(int i=0;i<maxIndex;i++)
	{
		*(labelToSet+i)=0;
	}
}
void setLabelFrequency(int *setLabelArray, int *labelToSet)
{
	initializeArray(labelToSet,255);
	int i=0;
	while(*(setLabelArray+i))
	{
		*(labelToSet+*(setLabelArray+i))=1;
		i++;
	}
}
int main()
{
	setLabelFrequency(setOfLabels,labelTotal);
	setLabelFrequency(setOfLabelsDigits,labelDigits);
	setLabelFrequency(setOfLabelsCalculatorSymbols ,labelCalculatorSymbols);
	setLabelFrequency(setOfLabelsCalculatorTotal,labelCalculatorTotal);
	setLabelFrequency(setOfLabelSlate,labelSlate);
	if(cap.isOpened()==false)
	{
		cout<<"Cannot open the video file"<<endl;
		cin.get();
		return -1;
	}
	int count=2;
	training();
	setIteratorValuesFromVectorElement(iteratorTrainedData,trainedData.begin(),trainedData.end());
	displayTrainedData();
	//namedWindow("original",WINDOW_NORMAL);
	namedWindow("red",WINDOW_NORMAL);
	namedWindow("canavas",WINDOW_NORMAL);
	getImageIntoMat(pathStrings+"slate.jpg",&heading);
	initializeMatObject(canavas);
	long delay=0;
	int flag=0;
	initializeMatObject(lastCanavas);
	vector <Mat> slateData;
	while(1)
	{
		bool bSuccess=cap.read(original);
		if(bSuccess==false)
		{
			break;
		}
		canavas=canavasBackup.clone();
		invertImage(original);
		split(original,spl);
		thresholding(canavas);
		canavasBackup=canavas.clone();
		Mat diff;
		bitwise_xor(lastCanavas,canavasBackup,diff);
		displayHeading(&canavas,5,320);
		int showX=10;
		for(vector <Mat> ::iterator it=slateData.begin();it!=slateData.end();it++)
		{
			displayImageAtPosition(&canavas,*it,100,showX);
	 		showX+=40;
		}
		imshow("red",spl[2]);
		imshow("canavas",canavas);
		int keyPress=waitKey(10);
		if(keyPress==27)//Esc key
		{
			initializeMatObject(canavas);
			initializeMatObject(canavasBackup);
			initializeMatObject(lastCanavas);
		}
		if(keyPress==13)//Enter key
		{
			initializeMatObject(resolved);
			int recognisedChar=recognition(canavasBackup,labelSlate);
			cout<<"Recognised Character = "<<(char)recognisedChar<<endl;
			initializeMatObject(canavas);
			initializeMatObject(canavasBackup);
			initializeMatObject(lastCanavas);
			if(recognisedChar)
			{
				
				slateData.push_back(((*(iteratorTrainedData+recognisedChar))->imageData).clone());
				secondActivity(recognisedChar);
				getImageIntoMat(pathStrings+"slate.jpg",&heading);
			    continue;
			}
			else
			{
				cout<<"Unable to recognise"<<endl;
			}
			// int ch='t';
			// if(writeImage(ch,count))
			// {
			// 	count++;
			// }
			// else
			// {
			// 	return 0;
			// }
			
 			/*Mat image = imread("/home/anubhaw/trainingImages/48_0.jpg");
 			namedWindow("new",WINDOW_NORMAL);
 			imshow("new",image);*/
			// while(1)
			// {
			// 	if(waitKey(10)==13)
			// 	{
			// 		initializeMatObject(canavas);
			// 		// destroyWindow("resolved");
			// 		// destroyWindow("scaled");
			// 		// destroyWindow("filtered");
			// 		break;
			// 	}
			// }
		}


	}
}