import argparse
from subprocess import Popen,PIPE
from numpy import trapz, mean, array, less, arange, full, where
from numpy import max as npmax
from lmfit.models import GaussianModel
from scipy.stats import pearsonr
from scipy.signal import argrelextrema
from sklearn import mixture

def fit_gaussian(y,x):    
    x=array(x)
    y=array(y)
    mod=GaussianModel()
    pars=mod.guess(y,x=x)
    result=mod.fit(y,pars,x=x)
    a=result.params['amplitude'].value
    b=result.params['center'].value
    c=result.params['sigma'].value
    best=result.best_fit
    chsqred=result.redchi
    chisq=result.chisqr
    fwhm=result.params['fwhm'].value
    
    return a,b,c,best,fwhm,chisq,chsqred

def area_under_curve(y_data): 
    area=trapz(y_data)
    
    return area   
    
def get_peak_stats(gaus_fit,area_calc):
    c=gaus_fit[2]
    best=gaus_fit[3]
    ci95_width=2*1.96*c
    fitted_height=max(best)-min(best)
    ht_to_wid=fitted_height/ci95_width
    
    return ci95_width,fitted_height,ht_to_wid

def unfold_bedgraph(coverage_from_query):
    unfolded_x=[]
    unfolded_y=[]
    for coverage_line in coverage_from_query:
        coverage_list=coverage_line.split()
        for i in range(int(coverage_list[2])-int(coverage_list[1])):
            unfolded_x.append(int(coverage_list[1])+1+i)
            unfolded_y.append(float(coverage_list[3]))
     
    return unfolded_x,unfolded_y

def get_minima(y_vector):
    min_indices=argrelextrema(y_vector,less)
    if len(min_indices[0])<2:
        trim_indices=0,len(y_vector)
    else:    
        trim_indices=min_indices[0][0],min_indices[0][-1]
        if trim_indices[1]-trim_indices[0]<100: # need to come up with a better solution 
            trim_indices=0,len(y_vector)
    
    return trim_indices

def plateauiness(y_vector,prop_max=0.80):
    plat_stat=sum(i>=npmax(y_vector)*prop_max for i in y_vector)
    
    return plat_stat

def eval_double_peak(x_vector,y_vector):
    x_vector=array(x_vector)
    y_vector=array(y_vector)
    g2 = mixture.GMM(n_components=2)
    g2.fit(y_vector)
    split=g2.predict(y_vector)
    print 'split: ',split
    x1=x_vector[where(split==1)]
    x2=x_vector[where(split==0)]
    y1=y_vector[where(split==1)]
    y2=y_vector[where(split==0)]
    
    g1 = mixture.GMM(n_components=1)
    g1.fit(y_vector)
    
    g2_bic=g2.bic(y_vector)
    g1_bic=g1.bic(y_vector)
    
    if g2_bic<g1_bic:
        return x1,y1,x2,y2,g1_bic,g2_bic
    else:
        return g1_bic,g2_bic    
    
    

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Arguments for reading bed,bedgraph, and bigwig for peak focusing')
    parser.add_argument('-covin','--coverage_infile',dest='infile',type=str,help='normalized coverage infile')
    parser.add_argument('-bg','--bedgraph',action='store_true',dest='bedgraph',help='infile is bedgraph format')
    parser.add_argument('-intv','--intervals_file',dest='intervals',help='intervals within which to fit gaussian curves')
    parser.add_argument('-nti','--number_training_intervals',type=int,dest='n_training_intervals',help='number of intervals to estimate starting parameters for gaussian')
    parser.add_argument('-o','--outfile',type=str,dest='focus_out',help='outfile name to dump summary statistics on peak focusing')
    parser.add_argument('-n','--background_regions',action='store_true',dest='bedgraph',help='infile is bedgraph format')
    parser.add_argument('-cvout','--curvefit_outfile',dest='curvout',type=str,help='name of bedfile to store coordinate, coverage, and fitted coverage data')
    parser.add_argument('-negcvout','--neg_curvefit_outfile',dest='neg_curvout',default="notwritten",type=str,help='name of bedfile to store coordinate, coverage, and fitted coverage data for negative fit')
    parser.add_argument('-rs','--rescale_doc',dest='rescale',default=1,type=int,help='1 /library size scalar, to multiply with depth of coverage to regenerated raw count scale for bed track')
    parser.add_argument('-lpad','--left_peak_padding',dest='lpp',default=20,type=int,help='for peak intervals,padding added to left side of peak region to improve gauss fit')
    parser.add_argument('-rpad','--right_peak_padding',dest='rpp',default=20,type=int,help='for peak intervals,padding added to right side of peak region to improve gauss fit')
   
    parser.add_argument('-fng','--fit_neg_gauss',action="store_true",dest='neg_gauss',help='boolean specification of whether to fit negative gaussian to intervals')
    
    
    opts = parser.parse_args()
    
    fout=open(opts.focus_out,'w')
    double_unfold_out=open('double_split_'+opts.focus_out,'w')
    cout=open(opts.curvout,'w')
    if opts.neg_gauss==True:
        neg_cout=open(opts.neg_curvout,'w')
    
    fout.write('chrom,start,stop,width_95CI,gauss_a,gauss_b,gauss_c,FWHM,ChiSqr,ChiSqrReduced,best_height,mean_doc,auc,htwidratio,data_fit_pearsonr,plateauiness,neg_width95CI,neg_a,neg_b,neg_c,neg_fwhm,neg_chisq_raw,neg_reduced_chisq,neg_fit_height,neg_area,neg_ht_to_wid_ratio,neg_pearson4,chisq_ratio,emptyflag,BIC1,BIC2\n')
    
    intervals_open=open(opts.intervals,'r')
    if opts.bedgraph:
        counter=0
        for interval in intervals_open:
            empty_flag="False"
            counter+=1
            if counter%10==0:
                print 'processing interval',counter
            interval_list=interval.strip().split()
            #print 'interval list:  ',interval_list
            chrom,start,stop=interval_list[0],max(1,int(interval_list[1])-opts.lpp),int(interval_list[2])+opts.rpp
            query="tabix %s %s:%s-%s" % (opts.infile,chrom,start,stop)
            #print 'query is:  ',query
            stdout,stderr=Popen(query,shell=True,stderr=PIPE,stdout=PIPE).communicate()
            #print 'stdout:  ',type(stdout)
            ### fill empty interval with uniform, small values
            if not stdout:
                x=arange(start,stop+1,1).tolist()
                y=full((1,stop-start+1),0.0000000001).tolist()[0]
                empty_flag="True"
                print 'x: ', x
                print 'y: ', y
            else:
                coverage_lines=stdout.split('\n')[:-1]
                x,y=unfold_bedgraph(coverage_lines)
            
            meancov=mean(y)
            #### gaussian on the original peak interval #####
            fitgaus=fit_gaussian(y,x)
            a,b,c,best,fwhm,chisq_raw,reduced_chisq=fitgaus
            area=area_under_curve(best)
            width95CI,fit_height,ht_to_wid_ratio=get_peak_stats(fitgaus,area)
            plateau_calc=plateauiness(best)
            
            normalized_y=1000*array(y)/float(npmax(y))
            norm_fitgaus=fit_gaussian(normalized_y,x)
            norm_a,norm_b,_norm_c,norm_best,norm_fwhm,norm_chisq_raw,norm_reduced_chisq=norm_fitgaus
            norm_area=area_under_curve(best)
            norm_width95CI,norm_fit_height,norm_ht_to_wid_ratio=get_peak_stats(norm_fitgaus,norm_area)
            #########
           
            ### evaluate possibility of double peaks
            double_check=eval_double_peak(x,y)
            if len(double_check)==2:
                bic1,bic2=double_check
            else:
                x_gaus1,y_gaus1,x_gaus2,y_gaus2,bic1,bic2=double_check
                meancov_1=mean(y_gaus1)
                meancov_2=mean(y_gaus2)
                ### 1st gaussian in mixture ###
                fitgaus_1=fit_gaussian(y_gaus1,x_gaus1)
                a1,b1,c1,best_1,fwhm_1,chisq_raw_1,reduced_chisq_1=fitgaus_1
                area_1=area_under_curve(best_1)
                width95CI_1,fit_height_1,ht_to_wid_ratio_1=get_peak_stats(fitgaus_1,area_1)
                plateau_calc_1=plateauiness(best_1)
            
                normalized_y_1=1000*array(y_gaus1)/float(npmax(y_gaus1))
                norm_fitgaus_1=fit_gaussian(normalized_y_1,x_gaus1)
                norm_a_1,norm_b_1,_norm_c_1,norm_best_1,norm_fwhm_1,norm_chisq_raw_1,norm_reduced_chisq_1=norm_fitgaus_1
                norm_area_1=area_under_curve(best_1)
                norm_width95CI_1,norm_fit_height_1,norm_ht_to_wid_ratio_1=get_peak_stats(norm_fitgaus_1,norm_area_1)
                
                ### 2nd gaussian in mixture ###
                
                fitgaus_2=fit_gaussian(y_gaus2,x_gaus2)
                a2,b2,c2,best_2,fwhm_2,chisq_raw_2,reduced_chisq_2=fitgaus_2
                area_2=area_under_curve(best_2)
                width95CI_2,fit_height_2,ht_to_wid_ratio_2=get_peak_stats(fitgaus_2,area_2)
                plateau_calc_2=plateauiness(best_2)
            
                normalized_y_2=1000*array(y_gaus2)/float(npmax(y_gaus2))
                norm_fitgaus_2=fit_gaussian(normalized_y_2,x_gaus2)
                norm_a_2,norm_b_2,_norm_c_2,norm_best_2,norm_fwhm_2,norm_chisq_raw_2,norm_reduced_chisq_2=norm_fitgaus_2
                norm_area_2=area_under_curve(best_2)
                norm_width95CI_2,norm_fit_height_2,norm_ht_to_wid_ratio_2=get_peak_stats(norm_fitgaus_2,norm_area_2)
               
                
                ### HOW MUCH DO I WANT TO WRITE TO FILES, AND HOW?????  
            
               
            if opts.neg_gauss==True:
                neg_y=-1*array(y)+max(y)
                neg_crops=get_minima(neg_y)
                neg_y=neg_y[neg_crops[0]:neg_crops[1]]
                neg_x=x[neg_crops[0]:neg_crops[1]]
                fit_neg_gaus=fit_gaussian(neg_y,neg_x)
                neg_a,neg_b,neg_c,neg_best,neg_fwhm,neg_chisq_raw,neg_reduced_chisq=fit_neg_gaus
                neg_area=area_under_curve(neg_best)
                neg_width95CI,neg_fit_height,neg_ht_to_wid_ratio=get_peak_stats(fit_neg_gaus,area)
            
                chisq_ratio=reduced_chisq/float(neg_reduced_chisq)
                neg_y_rescale=-1*neg_y+max(neg_y)
                neg_y_fit_rescale=-1*neg_best+max(neg_best)
                
                if len(double_check)==2: 
                    fout.write('region%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (counter,chrom,start,stop,width95CI,a,b,c,fwhm,chisq_raw,reduced_chisq,fit_height,meancov,area,ht_to_wid_ratio,pearsonr(y,best)[0],plateau_calc,neg_width95CI,neg_a,neg_b,neg_c,neg_fwhm,neg_chisq_raw,neg_reduced_chisq,neg_fit_height,neg_area,neg_ht_to_wid_ratio,pearsonr(neg_y,neg_best)[0],chisq_ratio,norm_fwhm,empty_flag,bic1,bic2))
                
                else:
                    fout.write('region%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (counter,chrom,start,stop,width95CI,a,b,c,fwhm,chisq_raw,reduced_chisq,fit_height,meancov,area,ht_to_wid_ratio,pearsonr(y,best)[0],plateau_calc,neg_width95CI,neg_a,neg_b,neg_c,neg_fwhm,neg_chisq_raw,neg_reduced_chisq,neg_fit_height,neg_area,neg_ht_to_wid_ratio,pearsonr(neg_y,neg_best)[0],chisq_ratio,norm_fwhm,empty_flag,bic1,bic2))
                    double_unfold_out.write('region%s_1,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (counter,chrom,x_gaus1[0],x_gaus1[-1],width95CI_1,a1,b1,c1,fwhm_1,chisq_raw_1,reduced_chisq_1,fit_height_1,meancov_1,area_1,ht_to_wid_ratio_1,pearsonr(y_gaus1,best_1)[0],plateau_calc_1,neg_width95CI,neg_a,neg_b,neg_c,neg_fwhm,neg_chisq_raw,neg_reduced_chisq,neg_fit_height,neg_area,neg_ht_to_wid_ratio,pearsonr(neg_y,neg_best)[0],chisq_ratio,norm_fwhm_1,empty_flag,bic1,bic2))
                    double_unfold_out.write('region%s_2,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (counter,chrom,x_gaus2[0],x_gaus2[-1],width95CI_2,a2,b2,c2,fwhm_2,chisq_raw_2,reduced_chisq_2,fit_height_2,meancov_2,area_2,ht_to_wid_ratio_2,pearsonr(y_gaus2,best_2)[0],plateau_calc_2,neg_width95CI,neg_a,neg_b,neg_c,neg_fwhm,neg_chisq_raw,neg_reduced_chisq,neg_fit_height,neg_area,neg_ht_to_wid_ratio,pearsonr(neg_y,neg_best)[0],chisq_ratio,norm_fwhm_2,empty_flag,bic1,bic2))
                
                for i in range(len(neg_y_fit_rescale)):
                    neg_cout.write('%s\t%s\t%s\t%s\t%s\tregion_%s\n' % (chrom,int(neg_x[i])-1,neg_x[i],neg_y_rescale[i]/float(opts.rescale),neg_y_fit_rescale[i]/float(opts.rescale),counter))
                        
            else:
                if len(double_check)==2:
                    fout.write('region%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,%s,%s\n' % (counter,chrom,start,stop,width95CI,a,b,c,fwhm,chisq_raw,reduced_chisq,fit_height,meancov,area,ht_to_wid_ratio,pearsonr(y,best)[0],plateau_calc,norm_fwhm,empty_flag,bic1,bic2))
                else:
                    fout.write('region%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,%s,%s\n' % (counter,chrom,start,stop,width95CI,a,b,c,fwhm,chisq_raw,reduced_chisq,fit_height,meancov,area,ht_to_wid_ratio,pearsonr(y,best)[0],plateau_calc,norm_fwhm,empty_flag,bic1,bic2))
                    double_unfold_out.write('region%s_1,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,%s,%s\n' % (counter,chrom,x_gaus1[0],x_gaus1[-1],width95CI_1,a1,b1,c1,fwhm_1,chisq_raw_1,reduced_chisq_1,fit_height_1,meancov_1,area_1,ht_to_wid_ratio_1,pearsonr(y_gaus1,best_1)[0],plateau_calc_1,norm_fwhm_1,empty_flag,bic1,bic2))
                    double_unfold_out.write('region%s_2,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,%s,%s\n' % (counter,chrom,x_gaus2[0],x_gaus2[-1],width95CI_2,a2,b2,c2,fwhm_2,chisq_raw_2,reduced_chisq_2,fit_height_2,meancov_2,area_2,ht_to_wid_ratio_2,pearsonr(y_gaus2,best_2)[0],plateau_calc_2,norm_fwhm_2,empty_flag,bic1,bic2))
                   
                
            for i in range(len(x)):
                cout.write('%s\t%s\t%s\t%s\t%s\tregion_%s\n' % (chrom,int(x[i])-1,x[i],y[i]/float(opts.rescale),best[i]/float(opts.rescale),counter))
            
            
            
    else:
        raise Exception("coverage file is not gzipped bedgraph")             
    
    fout.close()               
    cout.close()    
        

        
        
        