class Mod_depth:
    """
    class for calculating modulation depth 
    p1= paths of the images to be studied
    We read these images into arrays , if not in image use umode directly
    after that get the modes |u(r,0)| and |u(r,p)| by sqrt
    normalise them and subtract to get p/a * |v_0(r)|
    normalise this as well

    for N , sum all the cells of the initial image
    for p_a I get better results with sum of all cells of crb mode
    then for modulation depth proceed according to the formula

    functions :

    reading : In case u have an image to read

    umode: normalises a given image array and gives out |u| mode. Image is N|u|^2

    CRB: Returns the CRB mode of given two normalised |u|

    modulation : Calculates modulation depth of the given mask wrt the given two image array

    modscan (Idea) : Scans all possible combinations of calculating CRB mode and returns an array of mod. depths.

    """
    def __init__(self):
        pass

    
    def reading(self,p1):
        img_1= np.array(io.imread(p1, as_gray=True))
        return img_1

    
    def umode(self,img1):              #function for normalising the input image (taking sqrt of the image first)
        u1=img1**0.5                   #|u(r,p)|
        u1_norm=u1/np.linalg.norm(u1)  # using predefined L2 norm in numpy #L2 norm is sqrt (sum of squares of elements)
        return u1_norm

    
    def CRB(self,u1_norm,u2_norm):    #function to find CRB mode
        diff=u2_norm-u1_norm          #according to the theory approximation
        p_a=np.linalg.norm(diff)
        diff_norm=diff/p_a            # Normalising the CRB mode by the same stratergy above
        return diff_norm,p_a
    
    def modulation(self,mask,u1,u2):                     #Function for getting modulation depth
        u1_norm,u2_norm=self.umode(u1),self.umode(u2)    #|u(r,0)| and |u(r,p)| defining
        diff_norm,p_a=self.CRB(u1_norm,u2_norm)          #finding CRB mode
        N=np.sum(u1)                                     #defining N (not necessary, gets cancelled out)
        num=np.sum(u1_norm*diff_norm*mask)               
        den=(np.sum((u1_norm**2)*(mask**2)))    
        dep=((p_a*num)/(den**0.5)) 
        return np.abs(dep)
    def modscan(self,u):    # u is the array of all image arrays
        
        mod=np.zeros((len(u),len(u)))
        for i in tqdm(range(len(u)), desc="Processing outer loop"):   #running all possible combinations of calculating CRB modes
                    for j in range(i,len(u)):
                             if i!=j:
                                     a=mod_deps.umode(u[i])
                                     b=mod_deps.umode(u[j])
                                     c,p_a=mod_deps.CRB(a,b)
                                     depth=mod_deps.modulation(c,u[i],u[j])
                                     mod[i][j]=depth             #array of all mod. depths (ignores repeats and calculation of crb with same modes)
        return mod
        

        

        
                            
        
