
import joblib
import streamlit as st
import pandas as pd
import sklearn

Model= joblib.load("PF_Zomato_Model_Final.pkl")
Inputs= joblib.load("PF_Zomato_Inputs_Final.pkl")

def prediction(online_order, book_table, location,cost_two_people, listed_type,listed_city, Cuisines, Rest_Type):
    # cuisines_substrings=['Desserts', ' Mangalorean', 'Salad', ' Chettinad', ' Mediterranean', ' Jewish', 'Tea', ' Biryani', 'Seafood', 'Beverages', ' Chinese', ' American', ' French', ' Juices', ' Oriya', ' Bar Food', 'Momos', ' Assamese', ' Middle Eastern', 'Burmese', ' Japanese', ' Hyderabadi', ' Charcoal Chicken', 'Drinks Only', ' Indian', 'South Indian', 'Arabian', ' Pizza', 'Kashmiri', 'Bengali', 'Finger Food', ' Sri Lankan', ' North Eastern', ' Andhra', ' Sindhi', ' BBQ', 'South American', 'Cafe', 'African', ' Burger', ' Wraps', ' Cafe', ' Tex-Mex', ' South Indian', 'Italian', ' Afghan', ' Hot dogs', 'German', ' Belgian', ' Bubble Tea', ' Lucknowi', ' Momos', ' Nepalese', ' European', 'Modern Indian', ' Seafood', 'Lucknowi', 'Continental', 'Biryani', 'Australian', 'Iranian', 'Sushi', ' Thai', 'Singaporean', ' Tibetan', ' Fast Food', 'Portuguese', 'Thai', 'Nepalese', 'Assamese', 'Coffee', ' Maharashtrian', ' Sandwich', ' Kerala', 'Malaysian', ' Malwani', ' Coffee', ' Raw Meats', ' Rolls', ' Turkish', 'Turkish', 'Belgian', ' Beverages', ' Vietnamese', 'Healthy Food', 'Asian', ' Rajasthani', 'Ice Cream', 'Bihari', 'Hyderabadi', ' Singaporean', ' Awadhi', ' Burmese', 'Japanese', ' Mithai', 'Maharashtrian', ' Pan Asian', ' Roast Chicken', 'Steak', ' Modern Indian', 'Spanish', 'Mexican', ' African', 'Parsi', ' Italian', 'Sandwich', 'Indonesian', ' Afghani', 'Mithai', ' Desserts', ' Steak', ' Ice Cream', ' Kebab', ' Bihari', 'Oriya', 'Gujarati', 'Korean', ' Greek', ' Street Food', 'British', 'Bohri', ' South American', ' German', 'Mughlai', 'Street Food', 'Goan', ' Finger Food', ' Drinks Only', 'European', ' Lebanese', ' Iranian', 'Rolls', ' Mexican', 'Awadhi', ' Mongolian', 'Kerala', ' British', 'Pizza', 'Tamil', 'French', 'North Eastern', 'Middle Eastern', 'Bar Food', ' Grill', 'BBQ', ' Bakery', 'Kebab', 'American', 'Tibetan', ' Konkan', ' Arabian', ' Paan', ' Healthy Food', ' Asian', ' Malaysian', 'Burger', 'Chinese', ' Spanish', 'Lebanese', ' Naga', 'North Indian', ' Sushi', ' Mughlai', 'Mediterranean', 'Juices', ' Kashmiri', ' Parsi', 'Charcoal Chicken', 'Bakery', ' Salad', ' Goan', 'Roast Chicken', ' Korean', ' Cantonese', 'Andhra', 'Naga', ' North Indian', 'Fast Food', 'Mangalorean', 'Vietnamese', 'Rajasthani', ' Vegan', ' Gujarati', ' Indonesian', ' Tea', ' Bengali', 'Konkan', 'Chettinad']
    # rest_type_substrings=['Pop Up', ' Bar', ' Mess', 'Food Truck', ' Bakery', 'Dhaba', ' Pub', 'Confectionery', ' Quick Bites', 'Lounge', ' Microbrewery', ' Dessert Parlor', 'Takeaway', ' Casual Dining', ' Sweet Shop', 'Quick Bites', 'Bhojanalya', ' Food Court', 'Microbrewery', ' Meat Shop', 'Bakery', 'Delivery', 'Cafe', 'Casual Dining', 'Bar', ' Kiosk', ' Cafe', ' Irani Cafee', ' Delivery', 'Food Court', ' Lounge', 'Dessert Parlor', 'Club', 'Fine Dining', 'Mess', ' Beverage Shop', 'Pub', 'Sweet Shop', 'Kiosk']
    test_df=pd.DataFrame(columns=Inputs)
    test_df.at[0,'online_order']= online_order
    test_df.at[0,'book_table']= book_table
    test_df.at[0,'location']= location
    test_df.at[0,'cost_two_people']= cost_two_people
    test_df.at[0,'listed_type']=  listed_type
    test_df.at[0,'listed_city']=  listed_city
    test_df.at[0,'Cuisines']= Cuisines
    test_df.at[0,'Rest_Type']= Rest_Type
    test_df.at[0,'Desserts'] = test_df["Cuisines"].str.contains('Desserts', na=False, case=False).astype("int")
    test_df.at[0,' Mangalorean'] = test_df["Cuisines"].str.contains(' Mangalorean', na=False, case=False).astype("int")
    test_df.at[0,'Salad'] = test_df["Cuisines"].str.contains('Salad', na=False, case=False).astype("int")
    test_df.at[0,' Chettinad'] = test_df["Cuisines"].str.contains(' Chettinad', na=False, case=False).astype("int")
    test_df.at[0,' Mediterranean'] = test_df["Cuisines"].str.contains(' Mediterranean', na=False, case=False).astype("int")
    test_df.at[0,' Jewish'] = test_df["Cuisines"].str.contains(' Jewish', na=False, case=False).astype("int")
    test_df.at[0,' Biryani'] = test_df["Cuisines"].str.contains(' Biryani', na=False, case=False).astype("int")
    test_df.at[0,'Seafood'] = test_df["Cuisines"].str.contains('Seafood', na=False, case=False).astype("int")
    test_df.at[0,'Beverages'] = test_df["Cuisines"].str.contains('Beverages', na=False, case=False).astype("int")
    test_df.at[0,' Chinese'] = test_df["Cuisines"].str.contains(' Chinese', na=False, case=False).astype("int")
    test_df.at[0,' American'] = test_df["Cuisines"].str.contains(' American', na=False, case=False).astype("int")
    test_df.at[0,' French'] = test_df["Cuisines"].str.contains(' French', na=False, case=False).astype("int")
    test_df.at[0,' Juices'] = test_df["Cuisines"].str.contains(' Juices', na=False, case=False).astype("int")
    test_df.at[0,' Oriya'] = test_df["Cuisines"].str.contains(' Oriya', na=False, case=False).astype("int")
    test_df.at[0,' Bar Food'] = test_df["Cuisines"].str.contains(' Bar Food', na=False, case=False).astype("int")
    test_df.at[0,'Momos'] = test_df["Cuisines"].str.contains('Momos', na=False, case=False).astype("int")
    test_df.at[0,' Assamese'] = test_df["Cuisines"].str.contains(' Assamese', na=False, case=False).astype("int")
    test_df.at[0,' Middle Eastern'] = test_df["Cuisines"].str.contains(' Middle Eastern', na=False, case=False).astype("int")
    test_df.at[0,'Burmese'] = test_df["Cuisines"].str.contains('Burmese', na=False, case=False).astype("int")
    test_df.at[0,' Japanese'] = test_df["Cuisines"].str.contains(' Japanese', na=False, case=False).astype("int")
    test_df.at[0,' Hyderabadi'] = test_df["Cuisines"].str.contains(' Hyderabadi', na=False, case=False).astype("int")
    test_df.at[0,' Charcoal Chicken'] = test_df["Cuisines"].str.contains(' Charcoal Chicken', na=False, case=False).astype("int")
    test_df.at[0,'Drinks Only'] = test_df["Cuisines"].str.contains('Drinks Only', na=False, case=False).astype("int")
    test_df.at[0,' Indian'] = test_df["Cuisines"].str.contains(' Indian', na=False, case=False).astype("int")
    test_df.at[0,'South Indian'] = test_df["Cuisines"].str.contains('South Indian', na=False, case=False).astype("int")
    test_df.at[0,'Arabian'] = test_df["Cuisines"].str.contains('Arabian', na=False, case=False).astype("int")
    test_df.at[0,' Pizza'] = test_df["Cuisines"].str.contains(' Pizza', na=False, case=False).astype("int")
    test_df.at[0,'Kashmiri'] = test_df["Cuisines"].str.contains('Kashmiri', na=False, case=False).astype("int")
    test_df.at[0,'Bengali'] = test_df["Cuisines"].str.contains('Bengali', na=False, case=False).astype("int")
    test_df.at[0,'Finger Food'] = test_df["Cuisines"].str.contains('Finger Food', na=False, case=False).astype("int")
    test_df.at[0,' Sri Lankan'] = test_df["Cuisines"].str.contains(' Sri Lankan', na=False, case=False).astype("int")
    test_df.at[0,' North Eastern'] = test_df["Cuisines"].str.contains(' North Eastern', na=False, case=False).astype("int")
    test_df.at[0,' Andhra'] = test_df["Cuisines"].str.contains(' Andhra', na=False, case=False).astype("int")
    test_df.at[0,' Sindhi'] = test_df["Cuisines"].str.contains(' Sindhi', na=False, case=False).astype("int")
    test_df.at[0,' BBQ'] = test_df["Cuisines"].str.contains(' BBQ', na=False, case=False).astype("int")
    test_df.at[0,'South American'] = test_df["Cuisines"].str.contains('South American', na=False, case=False).astype("int")
    test_df.at[0,'Cafe'] = test_df["Cuisines"].str.contains('Cafe', na=False, case=False).astype("int")
    test_df.at[0,'African'] = test_df["Cuisines"].str.contains('African', na=False, case=False).astype("int")
    test_df.at[0,' Burger'] = test_df["Cuisines"].str.contains(' Burger', na=False, case=False).astype("int")
    test_df.at[0,' Wraps'] = test_df["Cuisines"].str.contains(' Wraps', na=False, case=False).astype("int")
    test_df.at[0,' Tex-Mex'] = test_df["Cuisines"].str.contains(' Tex-Mex', na=False, case=False).astype("int")
    test_df.at[0,' South Indian'] = test_df["Cuisines"].str.contains(' South Indian', na=False, case=False).astype("int")
    test_df.at[0,'Italian'] = test_df["Cuisines"].str.contains('Italian', na=False, case=False).astype("int")
    test_df.at[0,' Afghan'] = test_df["Cuisines"].str.contains(' Afghan', na=False, case=False).astype("int")
    test_df.at[0,' Hot dogs'] = test_df["Cuisines"].str.contains(' Hot dogs', na=False, case=False).astype("int")
    test_df.at[0,'German'] = test_df["Cuisines"].str.contains('German', na=False, case=False).astype("int")
    test_df.at[0,' Belgian'] = test_df["Cuisines"].str.contains(' Belgian', na=False, case=False).astype("int")
    test_df.at[0,' Lucknowi'] = test_df["Cuisines"].str.contains(' Lucknowi', na=False, case=False).astype("int")
    test_df.at[0,' Bubble '] = test_df["Cuisines"].str.contains(' Bubble ', na=False, case=False).astype("int")
    test_df.at[0,' Momos'] = test_df["Cuisines"].str.contains(' Momos', na=False, case=False).astype("int")
    test_df.at[0,' Nepalese'] = test_df["Cuisines"].str.contains(' Nepalese', na=False, case=False).astype("int")
    test_df.at[0,' European'] = test_df["Cuisines"].str.contains(' European', na=False, case=False).astype("int")
    test_df.at[0,'Modern Indian'] = test_df["Cuisines"].str.contains('Modern Indian', na=False, case=False).astype("int")
    test_df.at[0,' Seafood'] = test_df["Cuisines"].str.contains(' Seafood', na=False, case=False).astype("int")
    test_df.at[0,'Lucknowi'] = test_df["Cuisines"].str.contains('Lucknowi', na=False, case=False).astype("int")
    test_df.at[0,'Continental'] = test_df["Cuisines"].str.contains('Continental', na=False, case=False).astype("int")
    test_df.at[0,'Biryani'] = test_df["Cuisines"].str.contains('Biryani', na=False, case=False).astype("int")
    test_df.at[0,'Australian'] = test_df["Cuisines"].str.contains('Australian', na=False, case=False).astype("int")
    test_df.at[0,'Iranian'] = test_df["Cuisines"].str.contains('Iranian', na=False, case=False).astype("int")
    test_df.at[0,'Sushi'] = test_df["Cuisines"].str.contains('Sushi', na=False, case=False).astype("int")
    test_df.at[0,' Thai'] = test_df["Cuisines"].str.contains(' Thai', na=False, case=False).astype("int")
    test_df.at[0,'Singaporean'] = test_df["Cuisines"].str.contains('Singaporean', na=False, case=False).astype("int")
    test_df.at[0,' Tibetan'] = test_df["Cuisines"].str.contains(' Tibetan', na=False, case=False).astype("int")
    test_df.at[0,' Fast Food'] = test_df["Cuisines"].str.contains(' Fast Food', na=False, case=False).astype("int")
    test_df.at[0,'Portuguese'] = test_df["Cuisines"].str.contains('Portuguese', na=False, case=False).astype("int")
    test_df.at[0,'Thai'] = test_df["Cuisines"].str.contains('Thai', na=False, case=False).astype("int")
    test_df.at[0,'Nepalese'] = test_df["Cuisines"].str.contains('Nepalese', na=False, case=False).astype("int")
    test_df.at[0,'Assamese'] = test_df["Cuisines"].str.contains('Assamese', na=False, case=False).astype("int")
    test_df.at[0,'Coffee'] = test_df["Cuisines"].str.contains('Coffee', na=False, case=False).astype("int")
    test_df.at[0,' Maharashtrian'] = test_df["Cuisines"].str.contains(' Maharashtrian', na=False, case=False).astype("int")
    test_df.at[0,' Sandwich'] = test_df["Cuisines"].str.contains(' Sandwich', na=False, case=False).astype("int")
    test_df.at[0,' Kerala'] = test_df["Cuisines"].str.contains(' Kerala', na=False, case=False).astype("int")
    test_df.at[0,'Malaysian'] = test_df["Cuisines"].str.contains('Malaysian', na=False, case=False).astype("int")
    test_df.at[0,' Malwani'] = test_df["Cuisines"].str.contains(' Malwani', na=False, case=False).astype("int")
    test_df.at[0,' Coffee'] = test_df["Cuisines"].str.contains(' Coffee', na=False, case=False).astype("int")
    test_df.at[0,' Raw Meats'] = test_df["Cuisines"].str.contains(' Raw Meats', na=False, case=False).astype("int")
    test_df.at[0,' Rolls'] = test_df["Cuisines"].str.contains(' Rolls', na=False, case=False).astype("int")
    test_df.at[0,' Turkish'] = test_df["Cuisines"].str.contains(' Turkish', na=False, case=False).astype("int")
    test_df.at[0,'Turkish'] = test_df["Cuisines"].str.contains('Turkish', na=False, case=False).astype("int")
    test_df.at[0,'Belgian'] = test_df["Cuisines"].str.contains('Belgian', na=False, case=False).astype("int")
    test_df.at[0,' Beverages'] = test_df["Cuisines"].str.contains(' Beverages', na=False, case=False).astype("int")
    test_df.at[0,' Vietnamese'] = test_df["Cuisines"].str.contains(' Vietnamese', na=False, case=False).astype("int")
    test_df.at[0,'Healthy Food'] = test_df["Cuisines"].str.contains('Healthy Food', na=False, case=False).astype("int")
    test_df.at[0,'Asian'] = test_df["Cuisines"].str.contains('Asian', na=False, case=False).astype("int")
    test_df.at[0,' Rajasthani'] = test_df["Cuisines"].str.contains(' Rajasthani', na=False, case=False).astype("int")
    test_df.at[0,'Ice Cream'] = test_df["Cuisines"].str.contains('Ice Cream', na=False, case=False).astype("int")
    test_df.at[0,'Bihari'] = test_df["Cuisines"].str.contains('Bihari', na=False, case=False).astype("int")
    test_df.at[0,'Hyderabadi'] = test_df["Cuisines"].str.contains('Hyderabadi', na=False, case=False).astype("int")
    test_df.at[0,' Singaporean'] = test_df["Cuisines"].str.contains(' Singaporean', na=False, case=False).astype("int")
    test_df.at[0,' Awadhi'] = test_df["Cuisines"].str.contains(' Awadhi', na=False, case=False).astype("int")
    test_df.at[0,' Burmese'] = test_df["Cuisines"].str.contains(' Burmese', na=False, case=False).astype("int")
    test_df.at[0,'Japanese'] = test_df["Cuisines"].str.contains('Japanese', na=False, case=False).astype("int")
    test_df.at[0,' Mithai'] = test_df["Cuisines"].str.contains(' Mithai', na=False, case=False).astype("int")
    test_df.at[0,'Maharashtrian'] = test_df["Cuisines"].str.contains('Maharashtrian', na=False, case=False).astype("int")
    test_df.at[0,' Pan Asian'] = test_df["Cuisines"].str.contains(' Pan Asian', na=False, case=False).astype("int")
    test_df.at[0,' Roast Chicken'] = test_df["Cuisines"].str.contains(' Roast Chicken', na=False, case=False).astype("int")
    test_df.at[0,'Steak'] = test_df["Cuisines"].str.contains('Steak', na=False, case=False).astype("int")
    test_df.at[0,' Modern Indian'] = test_df["Cuisines"].str.contains(' Modern Indian', na=False, case=False).astype("int")
    test_df.at[0,'Spanish'] = test_df["Cuisines"].str.contains('Spanish', na=False, case=False).astype("int")
    test_df.at[0,'Mexican'] = test_df["Cuisines"].str.contains('Mexican', na=False, case=False).astype("int")
    test_df.at[0,' African'] = test_df["Cuisines"].str.contains(' African', na=False, case=False).astype("int")
    test_df.at[0,'Parsi'] = test_df["Cuisines"].str.contains('Parsi', na=False, case=False).astype("int")
    test_df.at[0,' Italian'] = test_df["Cuisines"].str.contains(' Italian', na=False, case=False).astype("int")
    test_df.at[0,'Sandwich'] = test_df["Cuisines"].str.contains('Sandwich', na=False, case=False).astype("int")
    test_df.at[0,'Indonesian'] = test_df["Cuisines"].str.contains('Indonesian', na=False, case=False).astype("int")
    test_df.at[0,' Afghani'] = test_df["Cuisines"].str.contains(' Afghani', na=False, case=False).astype("int")
    test_df.at[0,'Mithai'] = test_df["Cuisines"].str.contains('Mithai', na=False, case=False).astype("int")
    test_df.at[0,' Desserts'] = test_df["Cuisines"].str.contains(' Desserts', na=False, case=False).astype("int")
    test_df.at[0,' Steak'] = test_df["Cuisines"].str.contains(' Steak', na=False, case=False).astype("int")
    test_df.at[0,' Ice Cream'] = test_df["Cuisines"].str.contains(' Ice Cream', na=False, case=False).astype("int")
    test_df.at[0,' Kebab'] = test_df["Cuisines"].str.contains(' Kebab', na=False, case=False).astype("int")
    test_df.at[0,' Bihari'] = test_df["Cuisines"].str.contains(' Bihari', na=False, case=False).astype("int")
    test_df.at[0,'Oriya'] = test_df["Cuisines"].str.contains('Oriya', na=False, case=False).astype("int")
    test_df.at[0,'Gujarati'] = test_df["Cuisines"].str.contains('Gujarati', na=False, case=False).astype("int")
    test_df.at[0,'Korean'] = test_df["Cuisines"].str.contains('Korean', na=False, case=False).astype("int")
    test_df.at[0,' Greek'] = test_df["Cuisines"].str.contains(' Greek', na=False, case=False).astype("int")
    test_df.at[0,' Street Food'] = test_df["Cuisines"].str.contains(' Street Food', na=False, case=False).astype("int")
    test_df.at[0,'British'] = test_df["Cuisines"].str.contains('British', na=False, case=False).astype("int")
    test_df.at[0,'Bohri'] = test_df["Cuisines"].str.contains('Bohri', na=False, case=False).astype("int")
    test_df.at[0,' South American'] = test_df["Cuisines"].str.contains(' South American', na=False, case=False).astype("int")
    test_df.at[0,' German'] = test_df["Cuisines"].str.contains(' German', na=False, case=False).astype("int")
    test_df.at[0,'Mughlai'] = test_df["Cuisines"].str.contains('Mughlai', na=False, case=False).astype("int")
    test_df.at[0,'Street Food'] = test_df["Cuisines"].str.contains('Street Food', na=False, case=False).astype("int")
    test_df.at[0,'Goan'] = test_df["Cuisines"].str.contains('Goan', na=False, case=False).astype("int")
    test_df.at[0,' Finger Food'] = test_df["Cuisines"].str.contains(' Finger Food', na=False, case=False).astype("int")
    test_df.at[0,' Drinks Only'] = test_df["Cuisines"].str.contains(' Drinks Only', na=False, case=False).astype("int")
    test_df.at[0,'European'] = test_df["Cuisines"].str.contains('European', na=False, case=False).astype("int")
    test_df.at[0,' Lebanese'] = test_df["Cuisines"].str.contains(' Lebanese', na=False, case=False).astype("int")
    test_df.at[0,' Iranian'] = test_df["Cuisines"].str.contains(' Iranian', na=False, case=False).astype("int")
    test_df.at[0,'Rolls'] = test_df["Cuisines"].str.contains('Rolls', na=False, case=False).astype("int")
    test_df.at[0,' Mexican'] = test_df["Cuisines"].str.contains(' Mexican', na=False, case=False).astype("int")
    test_df.at[0,'Awadhi'] = test_df["Cuisines"].str.contains('Awadhi', na=False, case=False).astype("int")
    test_df.at[0,' Mongolian'] = test_df["Cuisines"].str.contains(' Mongolian', na=False, case=False).astype("int")
    test_df.at[0,'Kerala'] = test_df["Cuisines"].str.contains('Kerala', na=False, case=False).astype("int")
    test_df.at[0,' British'] = test_df["Cuisines"].str.contains(' British', na=False, case=False).astype("int")
    test_df.at[0,'Pizza'] = test_df["Cuisines"].str.contains('Pizza', na=False, case=False).astype("int")
    test_df.at[0,'Tamil'] = test_df["Cuisines"].str.contains('Tamil', na=False, case=False).astype("int")
    test_df.at[0,'French'] = test_df["Cuisines"].str.contains('French', na=False, case=False).astype("int")
    test_df.at[0,'North Eastern'] = test_df["Cuisines"].str.contains('North Eastern', na=False, case=False).astype("int")
    test_df.at[0,'Middle Eastern'] = test_df["Cuisines"].str.contains('Middle Eastern', na=False, case=False).astype("int")
    test_df.at[0,'Bar Food'] = test_df["Cuisines"].str.contains('Bar Food', na=False, case=False).astype("int")
    test_df.at[0,' Grill'] = test_df["Cuisines"].str.contains(' Grill', na=False, case=False).astype("int")
    test_df.at[0,'BBQ'] = test_df["Cuisines"].str.contains('BBQ', na=False, case=False).astype("int")
    test_df.at[0,' Bakery'] = test_df["Cuisines"].str.contains(' Bakery', na=False, case=False).astype("int")
    test_df.at[0,'Kebab'] = test_df["Cuisines"].str.contains('Kebab', na=False, case=False).astype("int")
    test_df.at[0,'American'] = test_df["Cuisines"].str.contains('American', na=False, case=False).astype("int")
    test_df.at[0,'Tibetan'] = test_df["Cuisines"].str.contains('Tibetan', na=False, case=False).astype("int")
    test_df.at[0,' Konkan'] = test_df["Cuisines"].str.contains(' Konkan', na=False, case=False).astype("int")
    test_df.at[0,' Arabian'] = test_df["Cuisines"].str.contains(' Arabian', na=False, case=False).astype("int")
    test_df.at[0,' Paan'] = test_df["Cuisines"].str.contains(' Paan', na=False, case=False).astype("int")
    test_df.at[0,' Healthy Food'] = test_df["Cuisines"].str.contains(' Healthy Food', na=False, case=False).astype("int")
    test_df.at[0,' Asian'] = test_df["Cuisines"].str.contains(' Asian', na=False, case=False).astype("int")
    test_df.at[0,' Malaysian'] = test_df["Cuisines"].str.contains(' Malaysian', na=False, case=False).astype("int")
    test_df.at[0,'Burger'] = test_df["Cuisines"].str.contains('Burger', na=False, case=False).astype("int")
    test_df.at[0,'Chinese'] = test_df["Cuisines"].str.contains('Chinese', na=False, case=False).astype("int")
    test_df.at[0,' Spanish'] = test_df["Cuisines"].str.contains(' Spanish', na=False, case=False).astype("int")
    test_df.at[0,'Lebanese'] = test_df["Cuisines"].str.contains('Lebanese', na=False, case=False).astype("int")
    test_df.at[0,' Naga'] = test_df["Cuisines"].str.contains(' Naga', na=False, case=False).astype("int")
    test_df.at[0,'North Indian'] = test_df["Cuisines"].str.contains('North Indian', na=False, case=False).astype("int")
    test_df.at[0,' Sushi'] = test_df["Cuisines"].str.contains(' Sushi', na=False, case=False).astype("int")
    test_df.at[0,' Mughlai'] = test_df["Cuisines"].str.contains(' Mughlai', na=False, case=False).astype("int")
    test_df.at[0,'Mediterranean'] = test_df["Cuisines"].str.contains('Mediterranean', na=False, case=False).astype("int")
    test_df.at[0,'Juices'] = test_df["Cuisines"].str.contains('Juices', na=False, case=False).astype("int")
    test_df.at[0,' Kashmiri'] = test_df["Cuisines"].str.contains(' Kashmiri', na=False, case=False).astype("int")
    test_df.at[0,' Parsi'] = test_df["Cuisines"].str.contains(' Parsi', na=False, case=False).astype("int")
    test_df.at[0,'Charcoal Chicken'] = test_df["Cuisines"].str.contains('Charcoal Chicken', na=False, case=False).astype("int")
    test_df.at[0,'Bakery'] = test_df["Cuisines"].str.contains('Bakery', na=False, case=False).astype("int")
    test_df.at[0,' Salad'] = test_df["Cuisines"].str.contains(' Salad', na=False, case=False).astype("int")
    test_df.at[0,' Goan'] = test_df["Cuisines"].str.contains(' Goan', na=False, case=False).astype("int")
    test_df.at[0,'Roast Chicken'] = test_df["Cuisines"].str.contains('Roast Chicken', na=False, case=False).astype("int")
    test_df.at[0,' Korean'] = test_df["Cuisines"].str.contains(' Korean', na=False, case=False).astype("int")
    test_df.at[0,' Cantonese'] = test_df["Cuisines"].str.contains(' Cantonese', na=False, case=False).astype("int")
    test_df.at[0,'Andhra'] = test_df["Cuisines"].str.contains('Andhra', na=False, case=False).astype("int")
    test_df.at[0,'Naga'] = test_df["Cuisines"].str.contains('Naga', na=False, case=False).astype("int")
    test_df.at[0,' North Indian'] = test_df["Cuisines"].str.contains(' North Indian', na=False, case=False).astype("int")
    test_df.at[0,'Fast Food'] = test_df["Cuisines"].str.contains('Fast Food', na=False, case=False).astype("int")
    test_df.at[0,'Mangalorean'] = test_df["Cuisines"].str.contains('Mangalorean', na=False, case=False).astype("int")
    test_df.at[0,'Vietnamese'] = test_df["Cuisines"].str.contains('Vietnamese', na=False, case=False).astype("int")
    test_df.at[0,'Rajasthani'] = test_df["Cuisines"].str.contains('Rajasthani', na=False, case=False).astype("int")
    test_df.at[0,' Vegan'] = test_df["Cuisines"].str.contains(' Vegan', na=False, case=False).astype("int")
    test_df.at[0,' Gujarati'] = test_df["Cuisines"].str.contains(' Gujarati', na=False, case=False).astype("int")
    test_df.at[0,' Indonesian'] = test_df["Cuisines"].str.contains(' Indonesian', na=False, case=False).astype("int")
    test_df.at[0,' Bengali'] = test_df["Cuisines"].str.contains(' Bengali', na=False, case=False).astype("int")
    test_df.at[0,'Konkan'] = test_df["Cuisines"].str.contains('Konkan', na=False, case=False).astype("int")
    test_df.at[0,'Chettinad'] = test_df["Cuisines"].str.contains('Chettinad', na=False, case=False).astype("int")
    test_df.at[0,'Pop Up'] = test_df["Rest_Type"].str.contains('Pop Up', na=False, case=False).astype("int")
    test_df.at[0,'Food Truck'] = test_df["Rest_Type"].str.contains('Food Truck', na=False, case=False).astype("int")
    test_df.at[0,'Dhaba'] = test_df["Rest_Type"].str.contains('Dhaba', na=False, case=False).astype("int")
    test_df.at[0,'Confectionery'] = test_df["Rest_Type"].str.contains('Confectionery', na=False, case=False).astype("int")
    test_df.at[0,'Lounge'] = test_df["Rest_Type"].str.contains('Lounge', na=False, case=False).astype("int")
    test_df.at[0,' Dessert Parlor'] = test_df["Rest_Type"].str.contains(' Dessert Parlor', na=False, case=False).astype("int")
    test_df.at[0,'Takeaway'] = test_df["Rest_Type"].str.contains('Takeaway', na=False, case=False).astype("int")
    test_df.at[0,' Casual Dining'] = test_df["Rest_Type"].str.contains(' Casual Dining', na=False, case=False).astype("int")
    test_df.at[0,'Quick Bites'] = test_df["Rest_Type"].str.contains('Quick Bites', na=False, case=False).astype("int")
    test_df.at[0,'Bhojanalya'] = test_df["Rest_Type"].str.contains('Bhojanalya', na=False, case=False).astype("int")
    test_df.at[0,'Microbrewery'] = test_df["Rest_Type"].str.contains('Microbrewery', na=False, case=False).astype("int")
    test_df.at[0,' Meat Shop'] = test_df["Rest_Type"].str.contains(' Meat Shop', na=False, case=False).astype("int")
    test_df.at[0,'Delivery'] = test_df["Rest_Type"].str.contains('Delivery', na=False, case=False).astype("int")
    test_df.at[0,'Casual Dining'] = test_df["Rest_Type"].str.contains('Casual Dining', na=False, case=False).astype("int")
    test_df.at[0,'Bar'] = test_df["Rest_Type"].str.contains('Bar', na=False, case=False).astype("int")
    test_df.at[0,' Irani Cafee'] = test_df["Rest_Type"].str.contains(' Irani Cafee', na=False, case=False).astype("int")
    test_df.at[0,'Food Court'] = test_df["Rest_Type"].str.contains('Food Court', na=False, case=False).astype("int")
    test_df.at[0,' Lounge'] = test_df["Rest_Type"].str.contains(' Lounge', na=False, case=False).astype("int")
    test_df.at[0,'Dessert Parlor'] = test_df["Rest_Type"].str.contains('Dessert Parlor', na=False, case=False).astype("int")
    test_df.at[0,'Club'] = test_df["Rest_Type"].str.contains('Club', na=False, case=False).astype("int")
    test_df.at[0,'Fine Dining'] = test_df["Rest_Type"].str.contains('Fine Dining', na=False, case=False).astype("int")
    test_df.at[0,'Mess'] = test_df["Rest_Type"].str.contains('Mess', na=False, case=False).astype("int")
    test_df.at[0,' Beverage Shop'] = test_df["Rest_Type"].str.contains(' Beverage Shop', na=False, case=False).astype("int")
    test_df.at[0,'Pub'] = test_df["Rest_Type"].str.contains('Pub', na=False, case=False).astype("int")
    test_df.at[0,'Sweet Shop'] = test_df["Rest_Type"].str.contains('Sweet Shop', na=False, case=False).astype("int")
    test_df.at[0,'Kiosk'] = test_df["Rest_Type"].str.contains('Kiosk', na=False, case=False).astype("int")
#     for cuisine in cuisines_substrings:
#         test_df.at[0,cuisine] = test_df["Cuisines"].str.contains(cuisine, na=False, case=False).astype("int")
#     for res_type in rest_type_substrings:
#         test_df.at[0,res_type] = test_df["Rest_Type"].str.contains(res_type, na=False, case=False).astype("int")
    
    test_df.drop('Cuisines',axis=1,inplace=True)
    test_df.drop('Rest_Type',axis=1,inplace=True)
    result= Model.predict(test_df)
    return result[0]

def main():
    
    ## Setting up the page title
    st.set_page_config(page_title= 'Zomato Restraunts Success Prediction')
    
     # Add a title in the middle of the page using Markdown and CSS
    st.markdown("<h1 style='text-align: center;text-decoration: underline;color:GoldenRod'>Zomato Restraunts Success Prediction</h1>", unsafe_allow_html=True)
          
    online_order=st.radio('Does Restraunt have online orders?', ['Yes', 'No'])

    book_table=st.radio('Does Restraunt have booking tables?', ['Yes', 'No'])

    
    location=st.selectbox('Where is the Restraunt?', ['Banashankari', 'Basavanagudi', 'Mysore Road', 'Jayanagar',
       'Kumaraswamy Layout', 'Rajarajeshwari Nagar', 'Vijay Nagar',
       'Uttarahalli', 'JP Nagar', 'South Bangalore', 'City Market',
       'Bannerghatta Road', 'BTM', 'Kanakapura Road', 'Bommanahalli',
       'Electronic City', 'Sarjapur Road', 'Wilson Garden',
       'Shanti Nagar', 'Koramangala 5th Block', 'Richmond Road', 'HSR',
       'Koramangala 7th Block', 'Bellandur', 'Marathahalli', 'Whitefield',
       'East Bangalore', 'Old Airport Road', 'Indiranagar',
       'Koramangala 1st Block', 'Frazer Town', 'MG Road', 'Brigade Road',
       'Lavelle Road', 'Church Street', 'Ulsoor', 'Residency Road',
       'Shivajinagar', 'Infantry Road', 'St. Marks Road',
       'Cunningham Road', 'Race Course Road', 'Commercial Street',
       'Vasanth Nagar', 'Domlur', 'Koramangala 8th Block', 'Ejipura',
       'Jeevan Bhima Nagar', 'Old Madras Road', 'Seshadripuram',
       'Kammanahalli', 'Koramangala 6th Block', 'Majestic',
       'Langford Town', 'Central Bangalore', 'Brookefield',
       'ITPL Main Road, Whitefield', 'Varthur Main Road, Whitefield',
       'Koramangala 2nd Block', 'Koramangala 3rd Block',
       'Koramangala 4th Block', 'Koramangala', 'Hosur Road', 'RT Nagar',
       'Banaswadi', 'North Bangalore', 'Nagawara', 'Hennur',
       'Kalyan Nagar', 'HBR Layout', 'Rammurthy Nagar', 'Thippasandra',
       'CV Raman Nagar', 'Kaggadasapura', 'Kengeri', 'Sankey Road',
       'Malleshwaram', 'Sanjay Nagar', 'Sadashiv Nagar',
       'Basaveshwara Nagar', 'Rajajinagar', 'Yeshwantpur', 'New BEL Road',
       'West Bangalore', 'Magadi Road', 'Yelahanka', 'Sahakara Nagar',
       'Jalahalli', 'Hebbal', 'Nagarbhavi', 'Peenya', 'KR Puram'])
        
    cost_two_people=st.number_input('What is the average cost for two people',min_value=10, max_value=10000, value=800,step=10)

    listed_type =st.selectbox('What is Restraunt List Type?', ['Buffet', 'Cafes', 'Delivery', 'Desserts', 'Dine-out','Drinks & nightlife', 'Pubs and bars'])

    listed_city =st.selectbox('Which city is the restraunt in?', ['Banashankari', 'Bannerghatta Road', 'Basavanagudi', 'Bellandur'
         ,'Brigade Road', 'Brookefield', 'BTM', 'Church Street', 'Electronic City'
         ,'Frazer Town', 'HSR', 'Indiranagar', 'Jayanagar', 'JP Nagar', 'Kalyan Nagar'
         ,'Kammanahalli', 'Koramangala 4th Block', 'Koramangala 5th Block'
         ,'Koramangala 6th Block', 'Koramangala 7th Block', 'Lavelle Road'
         ,'Malleshwaram', 'Marathahalli', 'MG Road', 'New BEL Road', 'Old Airport Road'
         ,'Rajajinagar', 'Residency Road', 'Sarjapur Road', 'Whitefield'])

    Cuisines = st.multiselect('Select Restraunt Cuisines:',['Desserts', ' Mangalorean', 'Salad', ' Chettinad', ' Mediterranean', ' Jewish', ' Biryani', 'Seafood', 'Beverages', ' Chinese', ' American', ' French', ' Juices', ' Oriya', ' Bar Food', 'Momos', ' Assamese', ' Middle Eastern', 'Burmese', ' Japanese', ' Hyderabadi', ' Charcoal Chicken', 'Drinks Only', ' Indian', 'South Indian', 'Arabian', ' Pizza', 'Kashmiri', 'Bengali', 'Finger Food', ' Sri Lankan', ' North Eastern', ' Andhra', ' Sindhi', ' BBQ', 'South American', 'Cafe', 'African', ' Burger', ' Wraps', ' Tex-Mex', ' South Indian', 'Italian', ' Afghan', ' Hot dogs', 'German', ' Belgian', ' Lucknowi', ' Bubble ', ' Momos', ' Nepalese', ' European', 'Modern Indian', ' Seafood', 'Lucknowi', 'Continental', 'Biryani', 'Australian', 'Iranian', 'Sushi', ' Thai', 'Singaporean', ' Tibetan', ' Fast Food', 'Portuguese', 'Thai', 'Nepalese', 'Assamese', 'Coffee', ' Maharashtrian', ' Sandwich', ' Kerala', 'Malaysian', ' Malwani', ' Coffee', ' Raw Meats', ' Rolls', ' Turkish', 'Turkish', 'Belgian', ' Beverages', ' Vietnamese', 'Healthy Food', 'Asian', ' Rajasthani', 'Ice Cream', 'Bihari', 'Hyderabadi', ' Singaporean', ' Awadhi', ' Burmese', 'Japanese', ' Mithai', 'Maharashtrian', ' Pan Asian', ' Roast Chicken', 'Steak', ' Modern Indian', 'Spanish', 'Mexican', ' African', 'Parsi', ' Italian', 'Sandwich', 'Indonesian', ' Afghani', 'Mithai', ' Desserts', ' Steak', ' Ice Cream', ' Kebab', ' Bihari', 'Oriya', 'Gujarati', 'Korean', ' Greek', ' Street Food', 'British', 'Bohri', ' South American', ' German', 'Mughlai', 'Street Food', 'Goan', ' Finger Food', ' Drinks Only', 'European', ' Lebanese', ' Iranian', 'Rolls', ' Mexican', 'Awadhi', ' Mongolian', 'Kerala', ' British', 'Pizza', 'Tamil', 'French', 'North Eastern', 'Middle Eastern', 'Bar Food', ' Grill', 'BBQ', ' Bakery', 'Kebab', 'American', 'Tibetan', ' Konkan', ' Arabian', ' Paan', ' Healthy Food', ' Asian', ' Malaysian', 'Burger', 'Chinese', ' Spanish', 'Lebanese', ' Naga', 'North Indian', ' Sushi', ' Mughlai', 'Mediterranean', 'Juices', ' Kashmiri', ' Parsi', 'Charcoal Chicken', 'Bakery', ' Salad', ' Goan', 'Roast Chicken', ' Korean', ' Cantonese', 'Andhra', 'Naga', ' North Indian', 'Fast Food', 'Mangalorean', 'Vietnamese', 'Rajasthani', ' Vegan', ' Gujarati', ' Indonesian', ' Bengali',  'Konkan', 'Chettinad'])

    Rest_Type = st.multiselect('Select Restraunt Type:',['Pop Up', 'Food Truck','Dhaba', 'Confectionery', 'Lounge', ' Dessert Parlor', 'Takeaway', ' Casual Dining','Quick Bites', 'Bhojanalya', 'Microbrewery', ' Meat Shop', 'Delivery', 'Casual Dining', 'Bar', ' Irani Cafee', 'Food Court', ' Lounge', 'Dessert Parlor', 'Club', 'Fine Dining', 'Mess', ' Beverage Shop', 'Pub', 'Sweet Shop', 'Kiosk'])
    
    if st.button('predict'):
        results= prediction(online_order, book_table, location,cost_two_people, listed_type,listed_city, Cuisines,Rest_Type)
        if int(results)==1:
            st.text(f"Congratulations your restraunt has big chance to succeed")
        else:
            st.text(f"Unfortunately your restraunt has big chance to fail")
    
if __name__ == '__main__':
    main()
