function out = animate(tout,xout)

params;

if length(tout)>1
    
   t = 0:0.02:tout(end);
   
   for n = 1:1:length(t)
       
        q1 = interp1(tout,xout(:,1),t(n),'PCHIP');
        q2 = interp1(tout,xout(:,2),t(n),'PCHIP');
        th1 = q1;
        th2 = q1 + q2;
        
        x0 = 0;
        y0 = 0;
        x1 = l1*sin(th1);
        x2 = x1 + l2*sin(th2);

        y1 = -l1*cos(th1);
        y2 = y1 - l2*cos(th2);
        
        xa = [x0,x1];
        ya = [y0,y1];
        
        xb = [x1,x2];
        yb = [y1,y2];
        
        if n==1
              p1 = plot(xa,ya,'r-','LineWidth',2);
              hold on;
              p2 = plot(xb,yb,'g-','LineWidth',2);
              axis image;
              hold on;
            
       
        axis([x0+[-2 2] y0+[-2 2]]);
        grid on;
        axis off;
        
        else
            set(p1,'Xdata',xa,'Ydata',ya);
            set(p2,'Xdata',xb,'Ydata',yb);
            axis([x0+[-2 2] y0+[-2 2]]);
 
            axis on;
            M(n) = getframe;
        end
        drawnow
   end
   out = xout(end,:);
   
end
        
        
        
        
        
