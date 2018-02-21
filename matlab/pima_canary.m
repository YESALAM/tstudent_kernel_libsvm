[label_vector,instance_matrix] = libsvmread('../../pima-5-fold/pima-5-1tra.dat_libsvm');
[testing_label_vector,testing_instance_matrix] = libsvmread('../../pima-5-fold/pima-5-1tst.dat_libsvm');

best_accuracy = 0 ;
best_c = 2^-18 ;
best_d = 1 ;

initVal = -18;
endVal = 50;
for d=1:6    
    for j = initVal:endVal
        c = 2^j;        
        
        model = svmtrain(label_vector,instance_matrix,sprintf('-s 0 -c %d -t 5 -r 1 -d %d -q',c,d));
        [predicted_label,accuracy,prob_estimates] = svmpredict(testing_label_vector,testing_instance_matrix,model,'-b 0 -q');
        a = accuracy(1,1);
        if(a>best_accuracy)
            best_accuracy = a ;
            best_c = c ;
            best_d = d ;
        end
        fprintf("c %d d %d accuracy %f \n",c,d,a);
    end
end

disp(best_accuracy);
disp(best_c);
disp(best_d);