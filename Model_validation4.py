"""
A simple script that demonstrates how we classify textual data with sklearn.
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.lda import LDA
from numpy import array
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from operator import itemgetter
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition.truncated_svd import TruncatedSVD   
from sklearn.linear_model import Perceptron

def readLexicon(file_name):
    lexicon = open(file_name)       
    build_lexicon = set()
    for line in lexicon:
        build_lexicon.add(line.strip())
    return build_lexicon
    
    lexicon.close()



#read the reviews and their polarities
def loadReviews(fname):
    
    reviews=[]
    polarities=[]
    print type(reviews)
    f=open(fname)
    for line in f:
        #line  = unicode(line, errors='replace')
        review,rating=line.strip().split('\t')    
        reviews.append(review.lower())    
        polarities.append(int(rating))
        print type(reviews)
    f.close()

    return reviews,polarities


def readReviews(fname):
    #file_write = open('output.txt','w')
    f=open(fname)    
    new_reviews=[]
    modified_reviews = []
    polarities=[]
    cnt=0    
    print 'looping'
    for line in f:
        cnt+=1
        if cnt%500==0:print cnt,'reviews'
        #print type(new_reviews)        
        #line  = unicode(line, errors='replace')
        custReviews,rating=line.strip().split('\t')  
        polarities.append(int(rating))
        new_reviews.append(custReviews.lower())
        
        review=custReviews.lower()        
        #print new_reviews        
        #for review in new_reviews:
            
            #print len(review),len(reviewWords##)
            #print review   
            
        if True:
            
            review = review.split()            
            
            marked = review[0]
                 
            for word in range(1,len(review)):                
                #print type(word)
                #next_word = review[word+1]
                #print review[next_word]
                                                
                if ((review[word-1]=='not' or review[word-1] =='never') and (review[word] in poslex or review[word] in neglex)):
                    marked+=review[word]                    
                    #print word   
                    #print review[next_word-1:next_word+1]
                    #print review[next_word-1]+review[next_word]
                    #review[next_word-1:next_word+1] = [(review[next_word-1]+review[next_word])]
#                    word_before =  str(word+review[next_word])
#                    word_after = str(word+review[next_word])
#                    str(review).replace(word_before, word_after)
#                    print str(review).replace(word_before, word_after)                    
                    #print word+' '+review[next_word]
                elif((review[word-1]=='high'and review[word-2] =='never' or review[word-2]=='not') and (review[word] in poslex)):
                    marked+=review[word]
                else:
                    marked+=' '+review[word]
            
         #modified_reviews.append    
        #print review           
            #print new_reviews[new_reviews.index(review)]#= review        
        #print new_reviews.index(review)        
        #print review[0:len(review)]
        modified_reviews.append(marked.strip())
        #7
        
        #print modified_reviews
            
#    for item in modified_reviews:
#        file_write.write(str(item)+'\n')
        
    return modified_reviews,polarities
    
    f.close()
    #file_write.close()        
                
        
    

poslex = readLexicon('positive-words.txt')
neglex = readLexicon('negative-words.txt')    
#rev_train,pol_train=readReviews('reviews.txt')
rev_train,pol_train=readReviews('reviews3-1.txt')

print 'loadind complete'
rev_test,pol_test=readReviews('C:/Users/Sathya/Documents/testFile4.txt')



#count the number of times each term appears in a document and transform each doc into a count vector
counter = CountVectorizer()
counts_train = counter.fit_transform(rev_train)
#print type(counts_train)

#transform the counts into the tfidfd format. http://en.wikipedia.org/wiki/Tf%E2%80%93idf
transformer = TfidfTransformer()
transformed_train = transformer.fit_transform(counts_train)

#apply the same transformation on the test datqs
counts_test=counter.transform(rev_test)
transformed_test=transformer.transform(counts_test)


#print transformed_test.size
#make ane empty model
#classifierNB=LogisticRegression(C=3.0) #0.75
classifierNB=LinearSVC(C=5.0) #0.75
#classifier= GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
#                             max_depth=1, random_state=0)
#classifier= RandomForestClassifier()
classifier=Perceptron(penalty=None, alpha=0.0001, fit_intercept=True, n_iter=5, shuffle=True, verbose=0, eta0=1.0, n_jobs=1, random_state=0, class_weight=None, warm_start=False)
#classifierKNN=KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None)
#classifierNB=MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
classifierSGDC=SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, n_iter=5, shuffle=True, verbose=0, epsilon=0.1, n_jobs=1, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=None, warm_start=False) #0.65
#classifier=LDA(n_components=None, priors=None, shrinkage=None,store_covariance=False, tol=0.0001)

#fit the model on the training data
classifier.fit(transformed_train,pol_train)
classifierSGDC.fit(transformed_train,pol_train)
classifierNB.fit(transformed_train,pol_train)
#get the accuracy on the test data
print ' lregACCURACY:\t',classifier.score(transformed_test,pol_test)
#0.866095238095
#print 'PREDICTED:\t',classifier.predict(transformed_test)
#print 'CORRECT:\t', array(pol_test)
logit_list = list(array(classifier.predict(transformed_test)))
#print logit_list
print 'sgdc ACCURACY:\t',classifierSGDC.score(transformed_test,pol_test)

#print 'PREDICTED:\t',classifierSGDC.predict(transformed_test)
#print 'CORRECT:\t', array(pol_test)
#print 'SGDC_List'
SGDC_list = list(array(classifier.predict(transformed_test)))
print ' nbACCURACY:\t',classifierNB.score(transformed_test,pol_test)

#print 'PREDICTED:\t',classifierNB.predict(transformed_test)
#print 'NB_List'
NB_list = list(array(classifier.predict(transformed_test)))
#print 'CORRECT:\t', array(pol_test)
#correct_list = list(array(pol_test))
result_set = logit_list

for i in range(0,len(logit_list)):
    if(logit_list[i]+SGDC_list[i]+NB_list[i] >= 2):
        result_set[i] = 1
        #print logit_list[i],SGDC_list[i],NB_list[i],result_set[i],correct_list[i]
    else:
        result_set[i] = 0
        #print logit_list[i],SGDC_list[i],NB_list[i],result_set[i],correct_list[i]

predicted_result = array(result_set)

#print 'Preficted', predicted_result
#print map(len,predicted_result)
#
#print 'ACCURACY_:\t',classifier.score(transformed_test,array(result_set))
#print sum(array(result_set))
#
#print 'ACCURACY:\t',classifierNB.score(transformed_test,array(result_set))
#print sum(array(result_set))
#
#print 'ACCURACY:\t',classifierKNN.score(transformed_test,array(result_set))
#print sum(array(result_set))
#
#score(transformed_test, array(result_set))

count = 0
transformed_test_list = list(pol_test) 
predicted_result_list = list(predicted_result)
#print type(predicted_result_list)
#print type(pol_test)
for i in range(0,len(pol_test)):
    #print transformed_test_list[i]
    #print predicted_result_list[i]
    if transformed_test_list[i] == predicted_result_list[i]:
        count = count +1
#print len(transformed_test)
#print count
accuracy = float(count)/float(len(transformed_test_list))
print'Voting accuracy:' ,accuracy
#for item in result_set:
#    print result_set[item]
