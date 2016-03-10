# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 18:22:05 2015

@author: Team 4
"""

import urllib2, re, time
browser = urllib2.build_opener()
browser.addheaders=[('User-agent', 'Mozilla/5.0')]

#User input for the city to scrape
city=raw_input("Please enter the city and state for example New York NY: ").split()

#Checking for sanity of the city and state name
if len(city)>2:
    City = city[0]+"+"+city[1]
    State= city[2]
else:
    City = city[0] 
    State= city[1]
    
#find the total number of restaurants in the given city
url = 'http://www.yelp.com/search?cflt=restaurants&find_loc='+City+'%2C+'+State+'%2C+USA#find_desc&start=0'
response = browser.open(url)
html = response.read()
restaurants = re.search('<span class="pagination-results-window">(.*?)</span>',html,re.S)
restMax = int(restaurants.group(1).strip()[16:])
restMaxPages = int(restaurants.group(1).strip()[16:])/10
if restMaxPages%10 > 0:
    restMaxPages = restMaxPages/10 + 1
    
#User input for number of restaurants and the number of reviews to scrape
City_1=City.replace('+',' ')
print "Total number of restaurants in the city "+City_1+": "+str(restMax)
noOfRestaurants=int(raw_input("Please enter the number of restaurants that has to be scraped: "))
if noOfRestaurants%10 > 0: #Checking for user input and deciding the number of pages to traverse for restaurants and reviews
    restPages=noOfRestaurants/10 + 1
else:
    restPages=noOfRestaurants/10
noOfReviews=int(raw_input("Please enter the number of reviews that has to be limited per restaurant: "))

#Write the Scrape details such as city and state name to the output file.
print City_1+" "+State+" "+"Number of Restaurants: "+str(noOfRestaurants)+" "+"Number of Reviews: "+str(noOfReviews)
fileWriterpos=open('yelp_'+City_1+'_'+State+'_Pos.txt','w')
fileWriterneg=open('yelp_'+City_1+'_'+State+'_Neg.txt','w')
#fileWriter.write(City_1+" "+State+" "+"Number of Restaurants: "+str(noOfRestaurants)+" "+"Number of Reviews: "+str(noOfReviews)+"\n")
#fileWriter.write("################################################################################"+"\n\n\n\n")
  
#Iterate through each page of the restaurant list page until all the restaurant links are gathered.
restPage=0
restaurantCount=1
try:
    while restPage<restPages:
        url = 'http://www.yelp.com/search?cflt=restaurants&find_loc='+City+'%2C+'+State+'%2C+USA&start='+str(restPage*10)
    
        response = browser.open(url)
    
        html = response.read()
    
        restNames = re.finditer('<a class="biz-name" href="(.*)" data-hovercard-id="(.*?)">(.*?)</a>',html)
     
        for name in restNames: 
            if len(name.group(1)) < 100:        
                if restaurantCount > noOfRestaurants:
                    break
                else:
                    restaurantName=name.group(1)
                    restaurantId=name.group(2)
                    restName=name.group(3)
                    startTime=time.clock()
                    print "Processing Restaurant: "+restName
                
                #Access the page that list the restaurants and the link to their review page.
                    revUrl = 'http://www.yelp.com'+restaurantName
                    revResponse = browser.open(revUrl)
                    revHtml = revResponse.read()
                    reviews = re.search('<div class="page-of-pages arrange_unit arrange_unit--fill">(.*?)</div>',revHtml,re.S)
                    revMaxPages = reviews.group(1)
                    revMaxPages = revMaxPages.strip()[10:]
                    revMaxPages = int(revMaxPages)
                    if restMaxPages%40 > 0:
                        restMaxPages = restMaxPages/40 + 1
                    if noOfReviews%40 > 0:                
                        revPages=noOfReviews/40 + 1
                    else:
                        revPages=noOfReviews/40
                
                    revPage=0 
                    reviewCount=1
                    while revPage < revPages:
                    #Access the page that list the reviews.
                        revUrl = 'http://www.yelp.com'+restaurantName+'?start='+str(revPage*40)+'&sort_by=rating_asc'
                        revResponse = browser.open(revUrl)
                        revHtml = revResponse.read()
                        
                        reviewData=re.finditer('<div class="review-wrapper">(.*?)<div class="review-footer clearfix">',revHtml,re.S)
                    #Write the review to a text file.
                        for reviewInfo in reviewData:
                            if reviewCount > noOfReviews:
                                break
                            else:
                                #date=re.search('<meta itemprop="datePublished" content="(.*?)"',reviewInfo.group(1),re.S)                        
                                rating=re.search('<meta itemprop="ratingValue" content="(.*?)"',reviewInfo.group(1),re.S)                        
                                review=re.search('<p itemprop="description" lang="en">(.*?)</p>',reviewInfo.group(1),re.S)
                                if rating.group(1) == '1.0' or rating.group(1) == '2.0':
                                    #fileWriter.write("Restaurant Id: "+restaurantId+"\n"+"Restaurant Name: "+restName+"\n"+"Date: "+date.group(1)+"\n"+"Rating: "+rating.group(1)+"\n"+"Review: "+review.group(1)+"\t"+"0"+"\n")
                                    fileWriterneg.write(review.group(1)+"\t"+"0"+"\n")
                                    #fileWriter.write("**********************************************************************************"+"\n\n")
                                elif rating.group(1) == '4.0' or rating.group(1) == '5.0':
                                    fileWriterpos.write(review.group(1)+"\t"+"1"+"\n")
                                reviewCount=reviewCount+1
                    
                        revPage=revPage + 1
               
                    print " Time Taken: "+str(time.clock()-startTime)
                    restaurantCount=restaurantCount+1
                
        restPage = restPage+1
    fileWriterpos.close()
    fileWriterneg.close()
except:
    time.sleep(120)