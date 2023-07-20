LSTM_Units=256
Max_Allowed_Sequence_Length=512

def Get_Corpus_Transcripts():
    import pyodbc
    DB = pyodbc.connect(r'Driver={SQL Server};Server=(local);Database=TrainingCorpus;Trusted_Connection=yes;',autocommit=True)
    DB_Link=DB.cursor()
    Transcripts_Rows=DB_Link.execute("Select Transcript_Lower_Punc_Only_Sentence_Boundary,TextIdxKey \
                                        From TedTalk \
                                        Where Transcript_Lower_Punc_Only_Sentence_Boundary Is Not Null").fetchall()
    DB_Link.close()
    DB.close()
    Transcripts=dict()
    for Row in Transcripts_Rows:
        Transcripts[Row[1]]=Row[0]
    return(Transcripts)


Transcripts=Get_Corpus_Transcripts()

def Build_Corpus_Dictionary(Transcripts):

    #Transcripts assumed to be list of lists of spoken words with only punctuation 
    #          being sentence ending punctuation, all of which are space separated 
    import pickle, os
    if not os.path.exists('Corpus_Token.pickle'):
        import collections,nltk
        Corpus_Tokenizer = nltk.tokenize.treebank.TreebankWordTokenizer()
        Corpus_Token_Frequency = collections.Counter()
        for Transcript in Transcripts.values():
            for Token in Corpus_Tokenizer.tokenize(Transcript):
                Corpus_Token_Frequency[Token.lower()] += 1
                
        Corpus_Token_Index=dict({0:'<PAD>',1:'<OOV>'})
        Token_Number=2 # most frequent word/token
        for Token in collections.OrderedDict(Corpus_Token_Frequency.most_common()).keys():
            Corpus_Token_Index[Token]=Token_Number
            Idx+=1
        with open('Corpus_Token.pickle','wb') as Save_Corpus_Token:
            pickle.dump(Corpus_Token_Index,Save_Corpus_Token)
    else:
        Corpus_Token_Index=pickle.load(open('Corpus_Token.pickle','rb'))
    return(Corpus_Token_Index)

Corpus_Token_Index=Build_Corpus_Dictionary(Transcripts)

def Build_GloVe_Encoded_Corpus(Corpus_Token_Index,Transcripts):
    import numpy,os,pickle
    #Associate Corpus token with GloVe vector
    if not os.path.exists('glove300.pickle'):
        with open('glove.42B.300d.txt', encoding="utf8") as GloVe:  #uncased
            GloVe_Corpus_Index = numpy.zeros(shape=(len(Corpus_Token_Index)+2,300),dtype=numpy.float32)
            for Token_Vector in GloVe:
                Token, *Vector = Token_Vector.split()
                if Token in Corpus_Token_Index:
                    GloVe_Corpus_Index[Corpus_Token_Index[Token]]=numpy.array(vector[-300:], dtype=numpy.float32)
            GloVe_Corpus_Index[1]=numpy.mean(GloVe_Corpus_Index[2:,],axis=0) #OOV gets average GloVe vector
        with open('glove300.pickle','wb') as Save_Glove:
            pickle.dump(GloVe_Corpus_Index,Save_Glove)
    else:
        GloVe_Corpus_Index=pickle.load(open('glove300.pickle','rb'))

    #Set all words in Corpus_Token_Index that aren't in GloVe to index 1 (OOV) so they can be given the average GloVe Vector
    for Token,Token_Number in Corpus_Token_Index.items():
        if numpy.sum(GloVe_Corpus_Index[Token_Number,:])==0:  ##Corpus_Token_Not_In_GloVe
            Corpus_Token_Index[Token]=1
    return(Corpus_Token_Index,GloVe_Corpus_Index)

Corpus_Token_Index,GloVe_Corpus_Index=Build_GloVe_Encoded_Corpus(Corpus_Token_Index,Transcripts)

def Training_Text_To_Sequences(Transcripts,Corpus_Token_Index,Max_Allowed_Sequence_Length):
    import nltk,numpy
    Tokenizer = nltk.tokenize.treebank.TreebankWordTokenizer()
    Transcripts_Labels_Tokens=dict()
    Longest_Sequence=0
    for Transcript_Id,Transcript in Transcripts.items():
        Labels=[];Tokens=[];Sequence_Index=0;
        Transcript_Subset_Id=0
        for Token in Tokenizer.tokenize(Transcript):
            if any(Character in Token for Character in ['.','?','!']) and Sequence_Index>0:
                if Sequence_Index-1<0 or Sequence_Index-1>=len(Labels):
                    print(Transcript_Id,Transcript_Subset_Id,Sequence_Index,Tokens,Labels)
                Labels[Sequence_Index-1]=2  # also should cover situation where sentence ends with 
                                            # multiple sentence ending tokens (e.g !?!?!?)
            else:
                if Sequence_Index==Max_Allowed_Sequence_Length: # output this portion of the transcript 
                                                                # and prepare for next transcript portion
                    Longest_Sequence==Max_Allowed_Sequence_Length
                    Transcripts_Labels_Tokens[Transcript_Id,Transcript_Subset_Id]=(Labels,Tokens)
                    Labels=[];Tokens=[];Sequence_Index=0;
                    Transcript_Subset_Id+=1
                Tokens.append(Corpus_Token_Index[Token.lower()] if Token.lower() in Corpus_Token_Index else 1) #Handle OOV token
                Labels.append(1)
                
                Sequence_Index+=1
        if Longest_Sequence!=Max_Allowed_Sequence_Length:
            Longest_Sequence=len(Labels)
        Transcripts_Labels_Tokens[Transcript_Id,Transcript_Subset_Id]=(Labels,Tokens)
    
    Padded_Transcripts_Labels=numpy.array([Labels+[0]*(Longest_Sequence-len(Labels)) 
                                           for Labels in [Label_Token[0] 
                                                          for Label_Token in Transcripts_Labels_Tokens.values()]])
    Padded_Transcripts_Integers=numpy.array([Tokens+[0]*(Longest_Sequence-len(Tokens)) 
                                           for Tokens in [Label_Token[1] 
                                                          for Label_Token in Transcripts_Labels_Tokens.values()]])
    
    return(Padded_Transcripts_Labels,Padded_Transcripts_Integers,Longest_Sequence,Transcripts_Labels_Tokens)

Transcripts_Labels_Array,Transcripts_Integers_Array,Longest_Sequence,_ = Training_Text_To_Sequences(
    Transcripts,
    Corpus_Token_Index,
    Max_Allowed_Sequence_Length)

def Build_Model(LSTM_Units,Longest_Sequence,GloVe_Corpus_Index,Corpus_Token_Index):
    import tensorflow
    Model=tensorflow.keras.models.Sequential(name='BiLSTM_GloVe_Model')
    Model.add(tensorflow.keras.Input(shape=(Longest_Sequence,), dtype='int32',name='Input'))
    Model.add(tensorflow.keras.layers.Embedding(input_dim=len(Corpus_Token_Index) + 2,
                                      output_dim=300,
                                      embeddings_initializer=tensorflow.keras.initializers.Constant(GloVe_Corpus_Index),
                                      input_length=Longest_Sequence,
                                      mask_zero=True,
                                      name='GloVe_300_Dim',
                                      trainable=False))
    Model.add(tensorflow.keras.layers.Bidirectional(layer=tensorflow.keras.layers.LSTM(units=LSTM_Units,
                                                                                       return_sequences=True,
                                                                                       activation="tanh",
                                                                                       recurrent_activation="sigmoid",
                                                                                       recurrent_dropout=0.0,
                                                                                       unroll=False,
                                                                                       use_bias=True
                                                                                      )
                                                    ,name='LSTM_'+str(LSTM_Units)+'_Seq_1'))
    Model.add(tensorflow.keras.layers.Bidirectional(layer=tensorflow.keras.layers.LSTM(units=LSTM_Units,
                                                                                       return_sequences=True,
                                                                                       activation="tanh",
                                                                                       recurrent_activation="sigmoid",
                                                                                       recurrent_dropout=0.0,
                                                                                       unroll=False,
                                                                                       use_bias=True
                                                                                      )
                                                    ,name='LSTM_'+str(LSTM_Units)+'_Seq_2'))
    Model.add(tensorflow.keras.layers.Bidirectional(layer=tensorflow.keras.layers.LSTM(units=LSTM_Units,
                                                                                       return_sequences=True,
                                                                                       activation="tanh",
                                                                                       recurrent_activation="sigmoid",
                                                                                       recurrent_dropout=0.0,
                                                                                       unroll=False,
                                                                                       use_bias=True
                                                                                      )
                                                    ,name='LSTM_'+str(LSTM_Units)+'_Seq_3'))
    Model.add(tensorflow.keras.layers.Dropout(rate=.1,name='Dropout_.1'))
    Model.add(tensorflow.keras.layers.Dense(units=3,
                                                                kernel_initializer='normal',
                                                                activation='sigmoid',
                                                                name='Dense'))
    Model.compile(loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(ignore_class=0,
                from_logits=False, reduction=tensorflow.keras.losses.Reduction.NONE
            ),optimizer='adam')
    Model.save_weights('Temp_Save_Weights.keras')
    print(Model.summary())
    return Model

Model=Build_Model(LSTM_Units,Longest_Sequence,GloVe_Corpus_Index,Corpus_Token_Index)

def Train_Model(Model,LSTM_Units,Transcripts_Integers_Array,Transcripts_Labels_Array,Longest_Sequence):
    import sklearn,statistics,numpy,itertools,math,tensorflow
    Best_Epochs_for_each_split = list()
    F1_for_each_split = list()
    LSTM_Units=256
    for Cross_Validation_Iteration,(train_index, test_index) in enumerate(
                sklearn.model_selection.KFold(n_splits=5,shuffle = True,random_state=42
                    ).split(Transcripts_Integers_Array,Transcripts_Labels_Array)):
        print('Iteration',Cross_Validation_Iteration+1,'of 5')
        Model.load_weights('Temp_Save_Weights.keras')
        Training_History=Model.fit(x=Transcripts_Integers_Array[train_index],
                           y=Transcripts_Labels_Array[train_index],
                           validation_data=(Transcripts_Integers_Array[test_index], Transcripts_Labels_Array[test_index]),
                           verbose=2,
                           epochs=40, #actual epochs may be reduced by EarlyStopping
                           steps_per_epoch = len(Transcripts_Labels_Array[train_index]) // 8,
                           validation_steps = len(Transcripts_Labels_Array[test_index]) // 8,
                           batch_size=8,  
                           callbacks=[tensorflow.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                                              min_delta=0.0001,
                                                                              patience=2,
                                                                              verbose=1,
                                                                              mode="min",
                                                                              restore_best_weights=False),
                                     tensorflow.keras.callbacks.ModelCheckpoint(
                                         filepath="Restore_Sentence_"+str(LSTM_Units)+"unit_Triple_BiLSTM_"\
                                             +str(Longest_Sequence)+"MaxToken_KFold_"+str(Cross_Validation_Iteration+1)+".keras",
                                         monitor='val_loss',
                                         save_weights_only=True,
                                         verbose=1,
                                         options = tensorflow.train.CheckpointOptions(experimental_enable_async_checkpoint=True),
                                         save_best_only=True,
                                         mode='min')])
        print('Model Fit Done')

        Best_Epochs_for_each_split.append(1+float(numpy.argmin(Training_History.history['val_loss'])))

        Predicted_Classifications = numpy.argmax(Model.predict(x=Transcripts_Integers_Array[test_index]), axis=-1)
        Predicted_Classifications_With_Padding_Info=list(zip(list(itertools.chain(*Predicted_Classifications.tolist())),
                                                             list(itertools.chain(*Transcripts_Integers_Array[test_index].tolist()))))
        True_Classifications_With_Padding_Info=list(zip(list(itertools.chain(*Transcripts_Labels_Array[test_index].tolist())),
                                                        list(itertools.chain(*Transcripts_Integers_Array[test_index].tolist()))))

        #Model may predict a non-pad is a pad, rare, but it happens so manually correct that until a better way is found
        F1=sklearn.metrics.f1_score(y_true=[Token_Label[0] 
                                                for Token_Label in True_Classifications_With_Padding_Info 
                                                                if Token_Label[1]!=0],
                                    y_pred=[Token_Label[0] if Token_Label[0]!=0 else 1 
                                                for Token_Label in Predicted_Classifications_With_Padding_Info 
                                                                if Token_Label[1]!=0])
        print(F1)
        F1_for_each_split.append(F1)


    print(F1_for_each_split)
    #Assuming F1 for each kfold split is similar take the epoch number from the best one, tr
    # and compute final fit model using all data
    Model.load_weights('Temp_Save_Weights.keras')
    Model.fit(x=Transcripts_Integers_Array,
              y=Transcripts_Labels_Array,
              epochs=math.ceil(statistics.mean(Best_Epochs_for_each_split)),
              batch_size=8,
              verbose=2,
              steps_per_epoch = len(Transcripts_Labels_Array) // 8
             )
    Model.save('Restore_Sentence_'+str(LSTM_Units)+'_unit_BiLSTM_'+str(Longest_Sequence)+'MaxToken.keras')

Train_Model(Model,LSTM_Units,Transcripts_Integers_Array,Transcripts_Labels_Array,Longest_Sequence)
