import numpy as np
import copy

def make_prior(GP,parameters,X,Y,prior_dis=None,scale=1,fun_name='nmll'):
    from catlearn.regression.gp_bv.bounds import Boundary_conditions
    prior_lp={}
    parameters_set=sorted(list(set(parameters)))
    if isinstance(scale,(float,int)):
        scale={para:scale for para in parameters_set}
    if prior_dis is None:
        prior_dis={para:Uniform_prior() for para in parameters_set}
    for para in parameters_set:
        para_count=parameters.count(para)
        if 'length'==para and 'Multi' in str(GP.kernel):
            if para_count<len(X[0]):
                para_count=len(X[0])
        bounds=Boundary_conditions(bound_type='educated',scale=scale[para])
        bounds=bounds.create(GP,X,Y,[para]*para_count,log=False,fun_name=fun_name)
        prior_lp[para]=np.array([copy.deepcopy(prior_dis[para]).min_max(b[0],b[1]) for b in bounds])
    return prior_lp

class Prior_distribution:
    def sum_table(self,x_grid):
        pdf_grid=self.pdf(x_grid)
        min_where=np.where(pdf_grid==np.min(pdf_grid))
        max_where=np.where(pdf_grid==np.max(pdf_grid))
        str1='Min=({},{}) ; Max=({},{})'.format(self.round_num(x_grid[min_where]),\
            self.round_num(pdf_grid[min_where]),self.round_num(x_grid[max_where]),self.round_num(pdf_grid[max_where]))
        mean_c=np.trapz(pdf_grid*x_grid,x=x_grid)
        var_c=np.trapz(pdf_grid*x_grid**2,x=x_grid)-mean_c**2
        str2='E[X]={} ; Var[X]={}'.format(self.round_num(mean_c),self.round_num(var_c))
        return str1,str2

    def round_num(self,val,r=2):
        'Round a value'
        if not isinstance(val,(float,int)):
            val=val.item(0)
        return np.format_float_scientific(val,unique=False, precision=r)


class Uniform_prior(Prior_distribution):
    def __init__(self,start=-1e6,end=1e6,prob=1):
        'Uniform distribution'
        self.start=start
        self.end=end
        self.prob=prob
        
    def pdf(self,x):
        'Probability density function'
        if isinstance(x,(float,int)):
            return self.prob if self.start<=x<=self.end else 0
        else:
            index=(self.start<=x)&(x<=self.end)
            value=np.zeros(len(x))
            value[index]=self.prob
            return value
    
    def deriv(self,x):
        'The derivative of the probability density function as respect to x'
        return 0*(x*0+1)
    
    def ln_pdf(self,x):
        'Log of probability density function'
        return np.log(self.prob)*(x*0+1)
    
    def ln_deriv(self,x):
        'The derivative of the log of the probability density function as respect to x'
        return 0*(x*0+1)
    
    def update(self,start=None,end=None,prob=None):
        'Update the parameters of distribution function'
        if start!=None:
            self.start=start
        if end!=None:
            self.end=end
        if prob!=None:
            self.prob=prob
        return self
            
    def mean_var(self,mean,var):
        'Obtain the parameters of the distribution function by the mean and variance values'
        std=np.sqrt(var)
        self.start=mean-4*std
        self.end=mean+4*std
        self.prob=1/(self.end-self.start)
        return self
    
    def min_max(self,min_v,max_v):
        'Obtain the parameters of the distribution function by the minimum and maximum values'
        self.start=min_v
        self.end=max_v
        self.prob=1/(self.end-self.start)
        return self
    
    def __repr__(self):
        return 'Uniform({},{})'.format(self.round_num(self.start),self.round_num(self.end))


class Normal_prior(Prior_distribution):
    def __init__(self,mu=0,std=1):
        'Normal distribution'
        self.mu=mu
        self.std=std
        
    def pdf(self,x):
        'Probability density function'
        return np.exp(-0.5*((x-self.mu)/self.std)**2)/(np.sqrt(2*np.pi)*self.std)
    
    def deriv(self,x):
        'The derivative of the probability density function as respect to x'
        return (-(x-self.mu)/self.std)*self.pdf(x)
    
    def ln_pdf(self,x):
        'Log of probability density function'
        return -np.log(self.std)-0.5*np.log(2*np.pi)-0.5*((x-self.mu)/self.std)**2
    
    def ln_deriv(self,x):
        'The derivative of the log of the probability density function as respect to x'
        return -(x-self.mu)/self.std**2
    
    def update(self,mu=None,std=None):
        'Update the parameters of distribution function'
        if mu!=None:
            self.mu=mu
        if std!=None:
            self.std=std
        return self
            
    def mean_var(self,mean,var):
        'Obtain the parameters of the distribution function by the mean and variance values'
        self.mu=mean
        self.std=np.sqrt(var)
        return self
    
    def min_max(self,min_v,max_v):
        'Obtain the parameters of the distribution function by the minimum and maximum values'
        self.mu=np.exp((np.log(max_v)+np.log(min_v))/2)
        #self.mu=np.mean([max_v,min_v])
        #self.std=np.sqrt(0.5*((max_v-self.mu)**2+(min_v-self.mu)**2))
        self.std=(max_v-self.mu)/3
        return self
    
    def __repr__(self):
        return 'N({},{})'.format(self.round_num(self.mu),self.round_num(self.std))


class Lognormal_prior(Prior_distribution):
    def __init__(self,mu=0,std=1):
        'Log-normal distribution'
        self.mu=mu
        self.std=std
        
    def pdf(self,x):
        'Probability density function'
        return np.exp(-0.5*((np.log(x)-self.mu)/self.std)**2)/(x*np.sqrt(2*np.pi)*self.std)
    
    def deriv(self,x):
        'The derivative of the probability density function as respect to x'
        return (-1/x-(x-self.mu)/self.std)*self.pdf(x)
    
    def ln_pdf(self,x):
        'Log of probability density function'
        return -0.5*((np.log(x)-self.mu)/self.std)**2-np.log(x*self.std)-0.5*np.log(2*np.pi)
    
    def ln_deriv(self,x):
        'The derivative of the log of the probability density function as respect to x'
        return -(1+(np.log(x)-self.mu)/self.std**2)/x
    
    def update(self,mu=None,std=None):
        'Update the parameters of distribution function'
        if mu!=None:
            self.mu=mu
        if std!=None:
            self.std=std
        return self
            
    def mean_var(self,mean,var):
        'Obtain the parameters of the distribution function by the mean and variance values'
        self.mu=np.log(mean**2/np.sqrt(mean**2+var))
        self.std=np.sqrt(np.log(1+var/mean**2))
        return self
    
    def min_max(self,min_v,max_v):
        'Obtain the parameters of the distribution function by the minimum and maximum values'
        self.std=(np.log(max_v)-np.log(min_v))/4
        self.mu=(np.log(max_v)+np.log(min_v))/2+self.std**2
        return self
    
    def __repr__(self):
        return 'Log-Normal({},{})'.format(self.round_num(self.mu),self.round_num(self.std))


class Gamma_prior(Prior_distribution):
    def __init__(self,a=0,b=1):
        'Gamma distribution'
        self.a=a
        self.b=b
        
    def pdf(self,x):
        'Probability density function'
        return (self.b**self.a)*(x**(self.a-1))*np.exp(-self.b*x)/np.math.factorial(self.a-1)
    
    def update(self,a=None,b=None):
        'Update the parameters of distribution function'
        if a!=None:
            self.a=a
        if b!=None:
            self.b=b
        return self
            
    def mean_var(self,mean,var):
        'Obtain the parameters of the distribution function by the mean and variance values'
        self.a=round(mean**2/var)
        if self.a==0:
            self.a=1
        self.b=np.sqrt(self.a/var)
        return self
    
    def __repr__(self):
        return 'Gamma({},{})'.format(self.round_num(self.a),self.round_num(self.b))


class Invgamma_prior(Prior_distribution):
    def __init__(self,a=0,b=1):
        'Inverse-Gamma distribution'
        self.a=a
        self.b=b
        
    def pdf(self,x):
        'Probability density function'
        return (self.b**self.a)*(x**(-self.a-1))*np.exp(-self.b/x)/np.math.factorial(self.a-1)
    
    def deriv(self,x):
        'The derivative of the probability density function as respect to x'
        return ((-self.a-1)/x+self.b/x**2)*self.pdf(x)
    
    def ln_pdf(self,x):
        'Log of probability density function'
        return self.a*np.log(self.b)-(self.a+1)*np.log(x)-(self.b/x)-np.log(np.math.factorial(self.a-1))
    
    def ln_deriv(self,x):
        'The derivative of the log of the probability density function as respect to x'
        return -(self.a+1)/x+(self.b/x**2)
    
    def update(self,a=None,b=None):
        'Update the parameters of distribution function'
        if a!=None:
            self.a=a
        if b!=None:
            self.b=b
        return self
            
    def mean_var(self,mean,var):
        'Obtain the parameters of the distribution function by the mean and variance values'
        self.a=round(mean**2/var+2)
        self.b=mean*(self.a-1)
        return self.a,self.b
    
    def min_max(self,min_v,max_v):
        'Obtain the parameters of the distribution function by the minimum and maximum values'
        mode=np.exp((np.log(max_v)+np.log(min_v))/2)
        self.a=round(mode**2/max_v**2+2)
        self.b=mode*(self.a+1)
        return self
    
    def __repr__(self):
        return 'Inv-Gamma({},{})'.format(self.round_num(self.a),self.round_num(self.b))


class Laplace_prior(Prior_distribution):
    def __init__(self,mu=0,b=1):
        'Laplace distribution'
        self.mu=mu
        self.b=b
        
    def pdf(self,x):
        'Probability density function'
        return np.exp(-abs(x-self.mu)/self.b)/(2*self.b)
    
    def update(self,mu=None,b=None):
        'Update the parameters of distribution function'
        if mu!=None:
            self.mu=mu
        if b!=None:
            self.b=b
        return self
            
    def mean_var(self,mean,var):
        'Obtain the parameters of the distribution function by the mean and variance values'
        self.mu=mean
        self.b=np.sqrt(var/2)
        return self
    
    def __repr__(self):
        return 'Laplace({},{})'.format(self.round_num(self.mu),self.round_num(self.b))


class Logistic_prior(Prior_distribution):
    def __init__(self,mu=0,s=1):
        'Logistic distribution'
        self.mu=mu
        self.s=s
        
    def pdf(self,x):
        'Probability density function'
        return np.exp(-(x-self.mu)/self.s)/(self.s*(1+np.exp(-(x-self.mu)/self.s))**2)
    
    def update(self,mu=None,s=None):
        'Update the parameters of distribution function'
        if mu!=None:
            self.mu=mu
        if s!=None:
            self.s=s
        return self
            
    def mean_var(self,mean,var):
        'Obtain the parameters of the distribution function by the mean and variance values'
        self.mu=mean
        self.s=np.sqrt(var*3)/np.pi
        return self
    
    def __repr__(self):
        return 'Logistic({},{})'.format(self.round_num(self.mu),self.round_num(self.s))


class Gen_normal_prior(Prior_distribution):
    def __init__(self,mu=0,s=1,v=2):
        'Generalized normal distribution'
        self.mu=mu
        self.s=s
        self.v=v
        
    def pdf(self,x):
        'Probability density function'
        return np.exp(-((x-self.mu)/self.s)**(2*self.v))/(0.9064*2*self.s)
    
    def deriv(self,x):
        'The derivative of the probability density function as respect to x'
        return (2*self.v)*(-((x-self.mu)/self.s)**(2*self.v-1))*self.pdf(x)
    
    def update(self,mu=None,s=None,v=None):
        'Update the parameters of distribution function'
        if mu!=None:
            self.mu=mu
        if s!=None:
            self.s=s
        if v!=None:
            self.v=v
        return self
            
    def mean_var(self,mean,var):
        'Obtain the parameters of the distribution function by the mean and variance values'
        self.mu=mean
        self.s=np.sqrt(3*var)
        return self
    
    def __repr__(self):
        return 'Generalized-normal({},{},{})'.format(self.round_num(self.mu),self.round_num(self.s),self.v)


class Gen_lognormal_prior(Prior_distribution):
    def __init__(self,mu=0,s=1,v=2):
        'Generalized log-normal distribution'
        self.mu=mu
        self.s=s
        self.v=v
        
    def pdf(self,x):
        'Probability density function'
        return np.exp(-((np.log(x)-self.mu)/self.s)**(2*self.v))/(0.9064*2*self.s)
    
    def deriv(self,x):
        'The derivative of the probability density function as respect to x'
        return (2*self.v)*(-((np.log(x)-self.mu)/(self.s*x))**(2*self.v-1))*self.pdf(x)
    
    def ln_pdf(self,x):
        'Log of probability density function'
        return -((np.log(x)-self.mu)/self.s)**(2*self.v)-np.log(0.9064*2*self.s)
    
    def ln_deriv(self,x):
        'The derivative of the log of the probability density function as respect to x'
        return (-2*self.v*((np.log(x)-self.mu)/self.s)**(2*self.v-1))/(x*self.s)
    
    def update(self,mu=None,s=None,v=None):
        'Update the parameters of distribution function'
        if mu!=None:
            self.mu=mu
        if s!=None:
            self.s=s
        if v!=None:
            self.v=v
        return self
            
    def mean_var(self,mean,var):
        'Obtain the parameters of the distribution function by the mean and variance values'
        self.mu=np.log(mean)
        self.s=np.sqrt(np.log(1+var/mean**2))
        return self
    
    def min_max(self,min_v,max_v):
        'Obtain the parameters of the distribution function by the minimum and maximum values'
        self.mu=(np.log(max_v)+np.log(min_v))/2
        self.s=(np.log(max_v)-self.mu)
        return self
    
    def __repr__(self):
        return 'Generalized-lognormal({},{},{})'.format(self.round_num(self.mu),self.round_num(self.s),self.v)

class Inverse_prior(Prior_distribution):
    def __init__(self):
        'Inverse distribution'
        
    def pdf(self,x):
        'Probability density function'
        return 1/x
    
    def deriv(self,x):
        'The derivative of the probability density function as respect to x'
        return -1/(x**2)
    
    def ln_pdf(self,x):
        'Log of probability density function'
        return -np.log(x)
    
    def ln_deriv(self,x):
        'The derivative of the log of the probability density function as respect to x'
        return -1/x
    
    def update(self,mu=None,std=None):
        'Update the parameters of distribution function'
        return self
            
    def mean_var(self,mean,var):
        'Obtain the parameters of the distribution function by the mean and variance values'
        return self
    
    def min_max(self,min_v,max_v):
        'Obtain the parameters of the distribution function by the minimum and maximum values'
        return self
    
    def __repr__(self):
        return 'Inverse'
