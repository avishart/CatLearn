import numpy as np

class Variable_Transformation:
    def __init__(self,hyper_var=None,**kwargs):
        " Make variable transformation of hyperparameters into an interval of (0,1)"
        if hyper_var is None:
            self.transf_para_no(['length','noise','prefactor'])
        else:
            self.hyper_var=hyper_var.copy()
    
    def transf_para(self,parameters,model,X,Y,use_bounds=True,s=0.14):
        " Make a dictionary of the parameters for the variable transformation "
        parameters_set=sorted(parameters)
        if use_bounds:
            self.transf_para_bounds(parameters,parameters_set,model,X,Y,s=s)
        else:
            self.transf_para_no(parameters_set)
        return self
    
    def transf_para_no(self,parameters_set):
        " Make a dictionary of the parameters for the variable transformation with no information "
        self.hyper_var={para:{'mean':np.array([0.0]),'std':np.array([4.51])} for para in parameters_set}
        return self
    
    def transf_para_bounds(self,parameters,parameters_set,model,X,Y,s=0.14):
        " Make a dictionary of the parameters for the variable transformation with information "
        from .educated import Educated_guess
        bounds=Educated_guess(prior=model.prior,kernel=model.kernel,parameters=parameters).bounds(X,Y,parameters)
        self.hyper_var={para:{'mean':np.nanmean(bounds[para],axis=1).reshape(-1),'std':s*(bounds[para][:,1]-bounds[para][:,0]).reshape(-1)} for para in parameters_set}
        return self
    
    def transform_t_to_theta(self,t):
        " Do a variable transformation "
        return {para:self.transform_t_to_hyper(np.array(t[para]),para) for para in t.keys()}

    def transform_t_to_hyper(self,t,para,i=0):
        " Use the variable transformation for one hyperparameter "
        return self.numeric_limits(self.hyper_var[para]['std'][i]*np.log(t/(1-t))+self.hyper_var[para]['mean'][i])

    def transform_theta_to_t(self,hyper):
        " Do an inverse variable transformation "
        return {para:self.transform_hyper_to_t(np.array(hyper[para]),para) for para in hyper.keys()}

    def transform_hyper_to_t(self,hyper,para,i=0):
        " Use the variable transformation for one hyperparameter into the variable parameter t "
        return 1.0/(1.0+np.exp(-(hyper-self.hyper_var[para]['mean'][i])/self.hyper_var[para]['std'][i]))

    def numeric_limits(self,array,dh=0.1*np.log(np.finfo(float).max)):
        " Replace hyperparameters if they are outside of the numeric limits in log-space "
        return np.where(-dh<array,np.where(array<dh,array,dh),-dh)

    def t_to_theta_lines(self,lines,parameters):
        " Calculate hyperparameter grid lines from t grid lines "
        parameters_set=sorted(list(set(parameters)))
        count_para=[0]*len(parameters_set)
        lines_new=[]
        for p,para in enumerate(parameters):
            i_para=parameters_set.index(para)
            i=count_para[i_para]
            lines_new.append(self.transform_t_to_hyper(np.array(lines[p]),para,i=i))
            count_para[i_para]+=1
        return np.array(lines_new)
    
    def copy(self):
        " Copy the object. "
        return self.__class__(hyper_var=self.hyper_var)
    
    def __repr__(self):
        return "Variable_Transformation(hyper_var={})".format(self.hyper_var)
    