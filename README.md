Final Project for Deep Learning Course for Ali Abuliel and Amar Habib Allah 
to run the peoject:
  1) connect to colab
  2) add code shell in colab 
  3)connect colab to you drive : 
                from google.colab import drive
                drive.mount("/content/gdrive")
  4) go to your files directory: cd gdrive/MyDrive
  5) git clone thee project
  6) cd Deep_learning_course
  7) run the project:
          1) Enter at the shell % run main.py -t -e -f
                -t menas train the model
                -e means test the model on the test set
                -f means try TFGSM attack on the trained model 
                * you have to train the model before test and TFGSM attack*
