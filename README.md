#   n l p _ e v e n t _ e x t r a c t i o n  
  
  
  
 # # #   �[��}T�N 
  
 # # # #   1 .   B E R T �����~ 
  
 ` ` `  
 c d   / e v e n t _ e x t r a c t i o n _ c o d e s  
  
 p y t h o n 3   u t i l s . p y  
  
 c d   . . /  
  
 p y t h o n 3   p r e p r o c e s s . p y   \  
 	 - - c o r p u s _ p a t h   c o r p o r a / e v e n t _ c o r p u s . t x t   \  
 	 - - v o c a b _ p a t h   m o d e l s / g o o g l e _ z h _ v o c a b . t x t   \  
 	 - - d a t a s e t _ p a t h   c o r p o r a / e v e n t _ d a t a s e t . p t   - - p r o c e s s e s _ n u m   8   - - t a r g e t   b e r t  
  
 C U D A _ V I S I B L E _ D E V I C E S = 6 , 7 , 8 , 9   p y t h o n 3   p r e t r a i n . p y   \  
 	 - - d a t a s e t _ p a t h   c o r p o r a / e v e n t _ d a t a s e t . p t   - - v o c a b _ p a t h   m o d e l s / g o o g l e _ z h _ v o c a b . t x t   \  
 	 - - p r e t r a i n e d _ m o d e l _ p a t h   m o d e l s / b e r t / g o o g l e _ z h _ m o d e l . b i n   \  
 	 - - o u t p u t _ m o d e l _ p a t h   m o d e l s / b e r t / e v e n t _ b e r t _ m o d e l . b i n   \  
 	 - - w o r l d _ s i z e   4   - - g p u _ r a n k s   0   1   2   3   \  
 	 - - t o t a l _ s t e p s   5 0 0 0   - - s a v e _ c h e c k p o i n t _ s t e p s   5 0 0 0   - - b a t c h _ s i z e   3 2   \  
 	 - - e m b e d d i n g   w o r d _ p o s _ s e g   - - e n c o d e r   t r a n s f o r m e r   - - m a s k   f u l l y _ v i s i b l e   - - t a r g e t   b e r t  
 ` ` `  
  
 �l������~�[b�vb e r t !j�WN}����c�YN��OX[0R` m o d e l s / b e r t / ` -N	�� 
 ` ` `  
 ���c�h t t p s : / / p a n . b a i d u . c o m / s / 1 u t C L n P 5 _ j n T m j h f 9 a i g H x w    
 �c�Sx�9 d 9 r  
 ` ` `  
  
 傁������~� ���HQN}�` g o o g l e _ z h _ m o d e l . b i n ` 0R` m o d e l s / b e r t / ` , N}����c�YN� 
  
 ` ` `  
 ���c�h t t p s : / / p a n . b a i d u . c o m / s / 1 E - c X E a V D P t H t f A d C 0 S 2 1 e A    
 �c�Sx�0 0 d d  
 ` ` `  
  
 # # # #   2 .   !j�W���~ 
  
 -   ` B a s e l i n e   ( B E R T - b a s e   +   l i n e a r ) `  
  
     R�N�N{|�WۏL�҉r��v�^Rh�l 
  
     ` ` `  
     C U D A _ V I S I B L E _ D E V I C E S = 6 , 7 , 8 , 9   p y t h o n 3   t r a i n . p y   - - l e a r n i n g _ r a t e   5 e - 5  
      
     #   b a s e l i n e _ c r i t e r i o n _ w e i g t h   =   [ 1 . 0 ,   0 . 0 ,   5 0 . 0 ] ;   1 0 0   e p o c h s ;   O p t i m i z e r :   l i n e a r .  
     C U D A _ V I S I B L E _ D E V I C E S = 6 , 7 , 8 , 9   n o h u p   p y t h o n 3   - u   t r a i n . p y   - - l e a r n i n g _ r a t e   5 e - 5   >   . . / l o g s / b a s e l i n e . o u t   2 > & 1   & !  
      
     ���� 
     C U D A _ V I S I B L E _ D E V I C E S = 6 , 7 , 8 , 9   p y t h o n 3   t r a i n . p y   \  
     	 - - m i d d l e _ m o d e l _ p a t h   . . / r e s u l t _ m o d e l s / b a s e l i n e / b e s t / m o d e l 0 . b i n  
     	  
     �~�g 
     E v e n t :   t o t a l _ r i g h t ,   t o t a l _ p r e d i c t ,   p r e d i c t _ r i g h t :   1 6 5 7 ,   1 7 7 1 ,   1 4 8 9  
     E v e n t :   p r e c i s i o n ,   r e c a l l ,   a n d   f 1 :   0 . 8 4 1 ,   0 . 8 9 9 ,   0 . 8 6 9  
     R o l e :   t o t a l _ r i g h t ,   t o t a l _ p r e d i c t ,   p r e d i c t _ r i g h t :   3 6 9 6 ,   4 0 0 5 ,   2 3 9 1  
     R o l e :   p r e c i s i o n ,   r e c a l l ,   a n d   f 1 :   0 . 5 9 7 ,   0 . 6 4 7 ,   0 . 6 2 1  
      
     #   b a s e l i n e _ c r i t e r i o n _ w e i g t h   =   [ 1 . 0 ,   0 . 0 ,   5 0 . 0 ] ;   2 0 0   e p o c h s ;   O p t i m i z e r :   p o l y n o m i a l .  
     C U D A _ V I S I B L E _ D E V I C E S = 4 , 5 , 6 , 7   n o h u p   p y t h o n 3   - u   t r a i n . p y   \  
     	 - - l e a r n i n g _ r a t e   5 e - 5   - - s c h e d u l e r   p o l y n o m i a l   - - e p o c h s _ n u m   2 0 0   \  
     	 - - s a v e _ d i r _ n a m e   b a s e l i n e - p o l y n o m i a l   \  
     	 >   . . / l o g s / b a s e l i n e - p o l y n o m i a l . o u t   2 > & 1   & !  
  
     C U D A _ V I S I B L E _ D E V I C E S = 4 , 5 , 6 , 7   n o h u p   p y t h o n 3   - u   t r a i n . p y   \  
     	 - - l e a r n i n g _ r a t e   5 e - 5   - - s c h e d u l e r   p o l y n o m i a l   - - e p o c h s _ n u m   2 0 0   \  
       	 - - m i d d l e _ m o d e l _ p a t h   . . / r e s u l t _ m o d e l s / b a s e l i n e - p o l y n o m i a l / b e s t / m o d e l 0 . b i n   \  
     	 - - s a v e _ d i r _ n a m e   b a s e l i n e - p o l y n o m i a l   \  
     	 >   . . / l o g s / b a s e l i n e - p o l y n o m i a l . o u t   2 > & 1   & !  
  
     E v e n t :   t o t a l _ r i g h t ,   t o t a l _ p r e d i c t ,   p r e d i c t _ r i g h t :   1 6 5 7 ,   1 7 2 9 ,   1 4 9 3  
     E v e n t :   p r e c i s i o n ,   r e c a l l ,   a n d   f 1 :   0 . 8 6 4 ,   0 . 9 0 1 ,   0 . 8 8 2  
     R o l e :   t o t a l _ r i g h t ,   t o t a l _ p r e d i c t ,   p r e d i c t _ r i g h t :   3 6 9 6 ,   3 8 6 5 ,   2 3 4 7  
     R o l e :   p r e c i s i o n ,   r e c a l l ,   a n d   f 1 :   0 . 6 0 7 ,   0 . 6 3 5 ,   0 . 6 2 1  
  
     #   b a s e l i n e _ c r i t e r i o n _ w e i g t h   =   [ 1 . 0 ,   0 . 0 ,   5 0 . 0 ] ;   4 0 0   e p o c h s ;   O p t i m i z e r :   p o l y n o m i a l .  
     C U D A _ V I S I B L E _ D E V I C E S = 4 , 5 , 6 , 7   n o h u p   p y t h o n 3   - u   t r a i n . p y   \  
     	 - - l e a r n i n g _ r a t e   5 e - 5   - - s c h e d u l e r   p o l y n o m i a l   - - e p o c h s _ n u m   4 0 0   \  
     	 - - s a v e _ d i r _ n a m e   b a s e l i n e - p o l y n o m i a l   \  
     	 >   . . / l o g s / b a s e l i n e - p o l y n o m i a l . o u t   2 > & 1   & !  
     ` ` `  
  
 -   ` B a s e l i n e - l s t m   ( B E R T - b a s e   +   L S T M   +   l i n e a r ) `  
  
     ` ` `  
     #   p t i m i z e r :   p o l y n o m i a l  
     C U D A _ V I S I B L E _ D E V I C E S = 6 , 7 , 8 , 9   p y t h o n 3   t r a i n . p y   \  
     	 - - l e a r n i n g _ r a t e   5 e - 5   - - m o d e l _ t y p e   b a s e l i n e - l s t m  
  
     C U D A _ V I S I B L E _ D E V I C E S = 6 , 7 , 8 , 9   n o h u p   p y t h o n 3   - u   t r a i n . p y   \  
     	 - - l e a r n i n g _ r a t e   5 e - 5   \  
     	 - - m o d e l _ t y p e   b a s e l i n e - l s t m   >   . . / l o g s / b a s e l i n e - l s t m . o u t   2 > & 1   & !  
     ` ` `  
      
 -   ` h i e r a r c h i c a l `  
  
     ` ` `  
     C U D A _ V I S I B L E _ D E V I C E S = 3   p y t h o n 3   t r a i n . p y   \  
     	 - - m o d e l _ t y p e   h i e r a r c h i c a l   \  
     	 - - s a v e _ d i r _ n a m e   h i e r a r a c h i c a l  
      
     #   1 0 0   e p o c h s ;   O p t i m i z e r :   l i n e a r ;   5   e v a l _ e p o c h s  
     C U D A _ V I S I B L E _ D E V I C E S = 3   n o h u p   p y t h o n 3   - u   t r a i n . p y   \  
     	 - - m o d e l _ t y p e   h i e r a r c h i c a l   \  
     	 - - e p o c h s _ n u m   1 0 0   - - e v a l _ s t e p s   5   \  
     	 - - s a v e _ d i r _ n a m e   h i e r a r a c h i c a l   >   . . / l o g s / h i e r a r c h i c a l . o u t   2 > & 1   & !  
     	  
     E v e n t :   t o t a l _ r i g h t ,   t o t a l _ p r e d i c t ,   p r e d i c t _ r i g h t :   1 6 5 7 ,   1 6 2 3 ,   1 4 6 7  
     E v e n t :   p r e c i s i o n ,   r e c a l l ,   a n d   f 1 :   0 . 9 0 4 ,   0 . 8 8 5 ,   0 . 8 9 5  
     R o l e :   t o t a l _ r i g h t ,   t o t a l _ p r e d i c t ,   p r e d i c t _ r i g h t :   3 6 9 6 ,   3 4 2 7 ,   2 3 4 3  
     R o l e :   p r e c i s i o n ,   r e c a l l ,   a n d   f 1 :   0 . 6 8 4 ,   0 . 6 3 4 ,   0 . 6 5 8  
     ` ` `  
  
 -   ` h i e r a r c h i c a l - b i a s `  
  
     ` ` `  
     #   1 0 0   e p o c h s ;   O p t i m i z e r :   l i n e a r ;   5   e v a l _ e p o c h s ;   b i a s - w e i g h t   1 0 . 0  
     C U D A _ V I S I B L E _ D E V I C E S = 3   n o h u p   p y t h o n 3   - u   t r a i n . p y   \  
     	 - - m o d e l _ t y p e   h i e r a r c h i c a l - b i a s   \  
     	 - - e p o c h s _ n u m   1 0 0   - - e v a l _ s t e p s   5   \  
     	 - - s a v e _ d i r _ n a m e   h i e r a r a c h i c a l - b i a s   >   . . / l o g s / h i e r a r c h i c a l - b i a s . o u t   2 > & 1   & !  
     	  
     #   1 0 0   e p o c h s ;   O p t i m i z e r :   p o l y n o m i a l ;   5   e v a l _ e p o c h s ;   b i a s - w e i g h t   1 0 . 0  
     C U D A _ V I S I B L E _ D E V I C E S = 3   n o h u p   p y t h o n 3   - u   t r a i n . p y   \  
     	 - - m o d e l _ t y p e   h i e r a r c h i c a l - b i a s   - - s c h e d u l e r   p o l y n o m i a l   \  
     	 - - e p o c h s _ n u m   1 0 0   - - e v a l _ s t e p s   5   \  
     	 - - s a v e _ d i r _ n a m e   h i e r a r a c h i c a l - b i a s   >   . . / l o g s / h i e r a r c h i c a l - b i a s . o u t   2 > & 1   & !  
     ` ` `  
      
 -   ` c a s c a d e `  
  
     ` ` `  
     C U D A _ V I S I B L E _ D E V I C E S = 2   p y t h o n 3   t r a i n . p y   \  
     	 - - m o d e l _ t y p e   c a s c a d e   \  
     	 - - e p o c h s _ n u m   1   - - e v a l _ s t e p s   1   \  
     	 - - s a v e _ d i r _ n a m e   c a s c a d e  
     ` ` `  
  
 -   ` c a s c a d e - b i a s `  
  
     ` ` `  
     C U D A _ V I S I B L E _ D E V I C E S = 2   p y t h o n 3   t r a i n . p y   \  
     	 - - m o d e l _ t y p e   c a s c a d e - b i a s   \  
     	 - - e p o c h s _ n u m   1   - - e v a l _ s t e p s   1   \  
     	 - - s a v e _ d i r _ n a m e   c a s c a d e - b i a s  
     	  
     C U D A _ V I S I B L E _ D E V I C E S = 2   n o h u p   p y t h o n 3   - u   t r a i n . p y   \  
     	 - - m o d e l _ t y p e   c a s c a d e - b i a s   \  
     	 - - e p o c h s _ n u m   1 0 0   - - e v a l _ s t e p s   5   \  
     	 - - s a v e _ d i r _ n a m e   c a s c a d e - b i a s   >   . . / l o g s / c a s c a d e - b i a s . o u t   2 > & 1   & !  
     ` ` `  
  
      
  
 