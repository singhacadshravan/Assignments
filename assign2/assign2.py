import math
from operator import itemgetter
def gettestdata():
	with open("test.txt") as ftest:
		#Content_list is the list that contains the read lines.     
		content_list = ftest.readlines()
		print(content_list)
	wordnum=0
	for line in content_list:
		wordnum+=1
		if wordnum%10000==0:
			print( "10000 more is done")
		tokens=list(line.split(" "))
		if len(tokens)==3:
			wordlisttest.append(tokens[0])
			poslisttest.append(tokens[1])
			ctaglisttest.append(tokens[2].strip())

	print( "List of POS tags=")
	print( len(poslisttest))
	print( "List of chunk tags")
	print( len(ctaglisttest))
	print( "Data loading is done")

def getpostags():
	with open("train.txt") as f:
		#Content_list is the list that contains the read lines.     
		content_list = f.readlines()
		print( content_list)
	wordnum=0
	for line in content_list:
		wordnum+=1
		if wordnum%10000==0:
			print( "10000 more is done")
		tokens=list(line.split(" "))
		if len(tokens)==3:
			wordlist.append(tokens[0])
			poslist.append(tokens[1])
			ctaglist.append(tokens[2].strip())
	print( "List of POS tags=")
	print( len(poslist))
	print( "List of chunk tags")
	print( len(ctaglist))
	print( "Data loading is done")

def features():
	for i in range(0,10):
		fpos[i]=[]
		ftag[i]=[]
	fpos[0].append(-1)
	ftag[0].append(0)
	
	ftag[1].append(-1)
	ftag[1].append(0)

	fpos[2].append(-2)
	ftag[2].append(0)

	ftag[3].append(-2)
	ftag[3].append(0)

	fpos[4].append(0)
	ftag[4].append(0)
	
	fpos[5].append(1)
	ftag[5].append(0)

	fpos[6].append(-2)
	fpos[6].append(0)
	ftag[6].append(0)

	fpos[7].append(-2)
	fpos[7].append(-1)
	ftag[7].append(0)

	ftag[8].append(-2)
	ftag[8].append(-1)
	fpos[8].append(0)

	fpos[9].append(-1)
	ftag[9].append(-1)
	fpos[9].append(0)
	ftag[9].append(0)

def evaluate(i):
	contexts=[]
	for index in range(3,len(wordlist)-1):
		contextkey=""
		for pos in fpos[i]:
			target=poslist[index+pos]
			contextkey=contextkey+"p"+str(pos)+"="+target+":"
		for tag in ftag[i]:
			target=ctaglist[index+tag]
			contextkey=contextkey+"t"+str(tag)+"="+target+":"
		if contextkey not in contexts:
			contexts.append(contextkey)
		if contextkey not in featurecount.keys():
			featurecount[contextkey]=1
		else:
			featurecount[contextkey]+=1
	counts=[]
	for context in contexts:
		counts.append((context,featurecount[context]))
	countsorted=sorted(counts,key=itemgetter(1))
	print("The top features are:")
	for i in range(0,3):
		print( "Index="+str(i))
		print(countsorted[len(counts)-i-1])
		if countsorted[len(counts)-i-1][0] not in finalfeatures.keys():
			finalfeatures[countsorted[len(counts)-i-1][0]]=countsorted[len(counts)-i-1][1]
	print("Printing feauture functions" )
	for k, v in finalfeatures.items():
		print(k,v)

def calfunc():
	for i in fpos.keys():
		evaluate(i)
def enumerate1(context,index):
	ctags=["t-1","t-2","t0","t-3"]
	pos=["p-1","p-2","p0","p-3"]
	flag=0
	chunkflag=0
	tokens=context.split(":")
	contextlength=len(tokens)-1
	for india in range(0,len(tokens)-1):
		posctags=tokens[india].split("=")
		if "t-1"==posctags[0]:
			if ctaglist[index-1]==posctags[1]:
				flag+=1
		if "t-2"==posctags[0]:
			if ctaglist[index-2]==posctags[1]:
				flag+=1
		if "t-3"==posctags[0]:
			if ctaglist[index-3]==posctags[1]:
				flag+=1
		if "t0"==posctags[0]:
			if ctaglist[index-0]==posctags[1]:
				flag+=1
		if "p-1"==posctags[0]:
			if poslist[index-1]==posctags[1]:
				flag+=1
		if "p-2"==posctags[0]:
			if poslist[index-2]==posctags[1]:
				flag+=1
		if "p-3"==posctags[0]:
			if poslist[index-3]==posctags[1]:
				flag+=1
		if "p0"==posctags[0]:
			if poslist[index-0]==posctags[1]:
				flag+=1
	if flag==contextlength:
		return 1
	else:
		return 0

def enumeratetest1(context,index):
	ctags=["t-1","t-2","t0","t-3"]
	pos=["p-1","p-2","p0","p-3"]
	flag=0
	chunkflag=0
	tokens=context.split(":")
	contextlength=len(tokens)-1
	for india in range(0,len(tokens)-1):
		posctags=tokens[india].split("=")
		if "t-1"==posctags[0]:
			if ctaglisttest[index-1]==posctags[1]:
				flag+=1
		if "t-2"==posctags[0]:
			if ctaglisttest[index-2]==posctags[1]:
				flag+=1
		if "t-3"==posctags[0]:
			if ctaglisttest[index-3]==posctags[1]:
				flag+=1
		if "t0"==posctags[0]:
			if ctaglisttest[index-0]==posctags[1]:
				flag+=1
		if "p-1"==posctags[0]:
			if poslisttest[index-1]==posctags[1]:
				flag+=1
		if "p-2"==posctags[0]:
			if poslisttest[index-2]==posctags[1]:
				flag+=1
		if "p-3"==posctags[0]:
			if poslisttest[index-3]==posctags[1]:
				flag+=1
		if "p0"==posctags[0]:
			if poslisttest[index-0]==posctags[1]:
				flag+=1
	if flag==contextlength:
		return 1
	else:
		return 0
def scgis1():
	observed={}
	expected={}
	delta={}
	lamdawgt={}
	Y=22
	I=len(poslist)
	s=[]
	for j in range(0,len(poslist)):
		s.append([0]*22)
	z=[]
	for i in range(0,len(poslist)):
		z.append(22)
	for cntxt in finalfeatures.keys():
		diff=3.0
		threshold=0.5
		noiter=0
		observed[cntxt]=finalfeatures[cntxt]
		delta[cntxt]=0
		lamdawgt[cntxt]=0
		expected[cntxt]=0		
		while diff>threshold:
			print("threshold gap="+str(diff))
			noiter+=1
			if noiter>5:
				break
			lamdaprev=lamdawgt[cntxt]
			for val in range(0,len(listoftags)):
				for index in range(3,len(wordlist)-1):
					str1=ctaglist[index].strip()
					str2=listoftags[val].strip()
					if str1!=str2:
						continue
					status=enumerate1(cntxt,index)
					if status==1:
						expected[cntxt]+=math.exp(s[index][val])/float(z[index])
			if expected[cntxt]==0:			
				qt=observed[cntxt]/(expected[cntxt]+0.001)
			else:
				qt=observed[cntxt]/(expected[cntxt])
			delta[cntxt]=math.log(qt)
			lamdawgt[cntxt]+=delta[cntxt]
			diff=abs(lamdawgt[cntxt]-lamdaprev)
			for val in range(0,len(listoftags)):
				for index in range(3,len(wordlist)-1):
					str3=ctaglist[index].strip()
					str4=listoftags[val].strip()
					if str3!=str4:
						continue
					status=enumerate1(cntxt,index)
					if status==1:
						z[index]=z[index]-math.exp(s[index][val])
						s[index][val]+=delta[cntxt]
						z[index]=z[index]+math.exp(s[index][val])

		print( "Context"+str(cntxt)+"is done in "+str(noiter)+"iterations")
	print("Value of lamdas learnt")
	fweights=open("featureweights.txt","w+")
	for k,v in lamdawgt.items():
		string1=str(k)+" "+str(v)
		fweights.write(string1)
		print( k,v)
	return lamdawgt

def test(lamdawgt):
	predictedtags=[]
	predictedtags.append('B-NP')
	predictedtags.append('B-NP')
	predictedtags.append('B-NP')
	for index in range(3,100):
		maxnum=0.0
		totaldenom=0.00
		for denomtag in range(0,len(listoftags)):
			featureval=0.00
			str3=ctaglisttest[index].strip()
			str4=listoftags[denomtag].strip()
			for cntxt in finalfeatures.keys():
				tokens=cntxt.split(":")
				flag=0
				#contextlength=len(tokens)-1
				for ind in range(0,len(tokens)-1):
					posctags=tokens[ind].split("=")
					if posctags[0]=="t0" and posctags[1]==str4:
						flag=1
				status=enumeratetest1(cntxt,index)
				if status==1:
					print( "Status="+str(status))
				print("Flag="+str(flag))
				if status==1 and flag==1:
					print("Printing loop info")
					print( "Current tag is="+str(str4))
					print( "Lamdawgt="+str(lamdawgt[cntxt]))
					featureval+=lamdawgt[cntxt]
			numerator=math.exp(featureval)
			if numerator>maxnum:
				maxnum=numerator
				chosentag=str4
			totaldenom+=math.exp(featureval)
		
		predictedtags.append(chosentag)
	for item in range(0,len(predictedtags)):
		print("Predictedtag="+str(predictedtags[item]))
		print("Actual Tag="+str(ctaglisttest[item]))

listoftags=['B-NP', 'B-PP', 'I-NP', 'B-VP', 'I-VP', 'B-SBAR', 'O', 'B-ADJP', 'B-ADVP', 'I-ADVP', 'I-ADJP', 'I-SBAR', 'I-PP', 'B-PRT', 'B-LST', 'B-INTJ', 'I-INTJ', 'B-CONJP', 'I-CONJP', 'I-PRT', 'B-UCP', 'I-UCP']
listofpos=['NN', 'IN', 'DT', 'VBZ', 'RB', 'VBN', 'TO', 'VB', 'JJ', 'NNS', 'NNP', ',', 'CC', 'POS', '.', 'VBP', 'VBG', 'PRP$', 'CD', '``', "''", 'VBD', 'EX', 'MD', '#', '(', '$', ')', 'NNPS', 'PRP', 'JJS', 'WP', 'RBR', 'JJR', 'WDT', 'WRB', 'RBS', 'PDT', 'RP', ':', 'FW', 'WP$', 'SYM', 'UH']
POS={}
TAG={}
fpos={}
ftag={}
fword={}
poslist=[]
ctaglist=[]
wordlist=[]
poslisttest=[]
ctaglisttest=[]
wordlisttest=[]
featurecount={}
finalfeatures={}
        
getpostags()
features()
calfunc()
print( "length of training data")
print(len(wordlist))
featurewgts=dict()
featurewgts=scgis1()   #responsible for training the wgts
gettestdata()
test(featurewgts)