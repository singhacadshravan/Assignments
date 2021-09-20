#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netdb.h>
#include <netinet/in.h>
#include <unistd.h>
#include <string.h>
#include <strings.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <fcntl.h> 
#include <errno.h>
#include <semaphore.h>
#include <sys/ipc.h> 
#include <sys/shm.h> 
#include <sys/prctl.h>
#include <signal.h>
#include <semaphore.h>
#include <pthread.h>

int processinput(char [100]);
int fillclientstruct(int );
void updateshmclient();
void sendwelcome( );
void sendclientdetails();
void handleclient();
void* deliveryprocess(void*);
void createstructure();
void* createshm(key_t ,int );
int deattachshm();
void exitprocess();
void sendkillmessage();
void updateshmclient();

void sendmessage(char [200]);

#define max 100
int clsockfd;
FILE *fp;       
sem_t *gbsemclient,*gbsemqueue;// global binary semaphore, client list and message queue
int gshmid[100],gshmcount; //for maintaing shmemory id so as to deallocate at exit
struct stclient
{ 
	int sockfd;
	int uniqueid;
	char uniquename[20];
} *client;
struct stmessage{
	char sendername[20];
	char receivername[20]; 
	int time; 
	char text[200];
	int deliverstatus; 
	int sockfd; 
};
struct stmessageq
{ 
	struct stmessage message[100];
	int head;
	int tail;
} *messageq;

void insertmessage(struct stmessage);
void writeinlog(struct stmessage );

int main(int argc, char *argv[])
{
	int sockfd,port=2000,pid;
	pthread_t rThread;
    char buffer[100],sendbuffer[10];
    struct sockaddr_in servaddr, cli_addr;
    signal(SIGINT, exitprocess);	
    
    memset(buffer,'\0',100);
    memset(sendbuffer,'\0',10);
    sockfd = socket(AF_INET, SOCK_STREAM, 0);   
    if(sockfd < 0)
    {
       perror("socket system call:");
       exit(-1);
    }
	bzero((char*)&servaddr,sizeof(servaddr));
	if(argc > 1)
	    port=atoi(argv[1]);        
	servaddr.sin_family = AF_INET;
	servaddr.sin_addr.s_addr = INADDR_ANY;
	servaddr.sin_port = htons(port);
	if(bind(sockfd,(struct sockaddr *)&servaddr, sizeof(servaddr)) < 0)
	{
		perror("bind system call");
		exit(-1);
	}
	if(listen(sockfd,10)==-1)
		perror("error in listening to port");
	int cliaddlen=sizeof(cli_addr);

	createstructure();// creating structure for shared memory for client and message queues
	
	int pstatus=pthread_create(&rThread,NULL,deliveryprocess,NULL);
	if (pstatus) {
		perror("ERROR: pthread_create():");
		exit(1);
	}
	else if (pid  < 0)perror("Forking child process failed:");	
	else
	{		
		while(1)
		{
			clsockfd = accept(sockfd,(struct sockaddr *)&cli_addr,&cliaddlen);
			if (clsockfd < 0)
			{
				perror("accept system call:");
				exit(-1);
			}
			if(fillclientstruct(clsockfd) == -1)
			{                    			    
    		    write(clsockfd,"Connection limit exceeded!! Only 5 clients can be connected.\n",100);
    		    fflush(clsockfd);
    		    fclose(clsockfd);
			}
			else 
		    {   if((pid=fork())==0) // if request from client is received
		        {	
		            //printf("before handleclient");
		            fflush(stdout);			
			        handleclient();
			        if (read(clsockfd,buffer,100 ) < 0)
			        {
				        perror("error in reading from socket");
				        exit(-1);
			        }		
			        write(clsockfd,sendbuffer,sizeof(sendbuffer));					           
		        }
		        else if (pid  < 0)perror("Forking child process failed:");	
		        else{
			        clsockfd=0;}
			}	    
		}
	}   
}
void insertmessage(struct stmessage tmessage)
{    
    if((messageq->head==0 && messageq->tail==max-1)||(messageq->head>0 && messageq->tail==messageq->head-1))
        printf("Message queue is overflowed\n");
    else
    {
        sem_wait(gbsemqueue);
        if(messageq->tail==max-1 && messageq->head>0)
        {
            messageq->tail=0;
            strcpy(messageq->message[messageq->tail].sendername,tmessage.sendername);
            strcpy(messageq->message[messageq->tail].receivername,tmessage.receivername);
            strncpy(messageq->message[messageq->tail].text,tmessage.text,200);
            messageq->message[messageq->tail].sockfd=tmessage.sockfd;
        }
        else // normal ones
        {
            if((messageq->head==0 && messageq->tail==-1) || (messageq->tail != messageq->head-1))
            {
                //messageq[++messageq->tail]=tmessage;
                strcpy(messageq->message[++messageq->tail].sendername,tmessage.sendername);
                strcpy(messageq->message[messageq->tail].receivername,tmessage.receivername);
                strncpy(messageq->message[messageq->tail].text,tmessage.text,200);
                messageq->message[messageq->tail].sockfd=tmessage.sockfd;
                
            }
        }
        sem_post(gbsemqueue);
    }
}

void handleclient()
{
    char buffer[200];
    memset(buffer,'\0',sizeof(buffer));
    int status;
    sendwelcome();        
    while(1)
    {
        status=read(clsockfd,buffer,200);
        if(status < 0)
		{
	        perror("error in reading from socket");
	        exit(-1);
		}
		else if(status!=0)
	    {
	        if(strncmp(buffer,"KILL",4)==0) // client is killing itself
	        {
	            sendkillmessage();    
	            updateshmclient();// these function need not be in sempost and wait
	        }
	        else if(strncmp(buffer,"details",7)==0) // client asks for details of other client
	            sendclientdetails();    
	        else
	        {
	            sendmessage(buffer);
	        }
	    }
	    else {} 	    
	}	
}
void sendmessage(char *buffer)
{
    //if string before ; is more than 30 then show error and after ; more than 5 then show error
    char treceiver[20];
    int index;
    struct stmessage tmessage;
    memset(&tmessage,'\0',sizeof tmessage);
    char *pointer = strchr(buffer, ':');
    //printf("sendmessage:%s\n",pointer);
    if(pointer==NULL)
    {
        write(clsockfd,"No sender is specified. sender name/id should be before :\n",100);
        return;
    }
    index = (int)(pointer - buffer);    
    strncpy(treceiver,buffer,index);
    //now check if receiver exist in list    
    int i,status=0;
    for(i=0;i<5;i++)
    {
		if(strcmp((client+i)->uniquename,treceiver)==0 || ((client+i)->uniqueid!=0 && (client+i)->uniqueid == atoi(treceiver) ) )
		{	
   		    strcpy(tmessage.receivername,(client+i)->uniquename);
   		    tmessage.sockfd=(client+i)->sockfd; // receiver sockfd   		    
   		    strncpy(tmessage.text,pointer+1,200);
		    status=1;		    
		}
		else if((client+i)->sockfd == clsockfd ) // to get the name of sender
		    strcpy(tmessage.sendername,(client+i)->uniquename);
		else{}   
	}
	if(status==0) // no match found in the client list and so send back the sender an error message
    {
        write(clsockfd,"No such receiver is online. Please check the list of online clients!\n",100);        
        return;
    }
    insertmessage(tmessage);   
}

// process that manage the delivery of message to the receiver and delete it from queue
void*  deliveryprocess(void * x)
{ 
    int index=-1;
    char text[220];
    while(1)
    {
        sem_wait(gbsemqueue);
        if((messageq->head==0)&&(messageq->tail==-1))
        {
            //printf("Queue is underflow\n");        
        }        
        else if(messageq->head==messageq->tail)
        {
            index= messageq->head;        
            messageq->tail=-1;
            messageq->head=0;
        }
        else
            if(messageq->head==max-1)
            {
                index= messageq->head;            
                messageq->head=0;
            }
            else 
            {
                index= messageq->head;            
                messageq->head++;
            }
        sem_post(gbsemqueue);
        
        if(index!=-1)
        {
            sprintf(text,"%s:%s",messageq->message[index].sendername,messageq->message[index].text);
            write(messageq->message[index].sockfd,text,sizeof(text));        
            writeinlog(messageq->message[index]);
            //memset(&(messageq->message[index]),'\0',sizeof(struct stmessage)); // cleaning the data for further use
        }
        index=-1;
    }    
}
void writeinlog(struct stmessage tmessage)
{
    //only first time create the file rest only writing 
    if(fp==NULL)
    {
        fp = fopen("log.csv", "w");
        if (fp == NULL) 
        {
            perror("error in opening file log.csv\n");
            exit(0);
        }
    }    
    int i;   
    fprintf(fp, "%s,%d,%s,%s", tmessage.sendername,strlen(tmessage.text),tmessage.receivername,tmessage.text);    
}

int fillclientstruct(int sockfd)
{
//	sem_wait(gbsemclient);
	static int idcount=10000;
	int i;
	for( i=0;i<5;i++)
		if((client+i)->uniqueid ==0) 
		{
			(client+i)->uniqueid=idcount;
			sprintf((client+i)->uniquename,"client%d",idcount-10000);
			(client+i)->sockfd=sockfd;            
			idcount++;
			break;					
		}
	if(i==5)// client limit exceeds	
        return -1;	    
//	sem_post(gbsemclient);
}
// when client is died then update client data structure
void updateshmclient()
{
    int i;
    for(i=0;i<5;i++)
    {
		if((client+i)->sockfd == clsockfd) 
		{
		    (client+i)->sockfd=0;
		    (client+i)->uniqueid=0;
		    memset((client+i)->uniquename,'\0',20);	    		    
		    break;
		}					
	}  
}
void sendwelcome()
{
    char buffer[200];int i;
    for(i=0;i<5;i++)
    {
		if((client+i)->sockfd == clsockfd) 
		{
		    sprintf(buffer,"Welcome Uniquename:%s Uniqueid:%d\n",(client+i)->uniquename,(client+i)->uniqueid);
		    write(clsockfd,buffer,sizeof(buffer));		    
		    break;
		}					
	}
}
void sendclientdetails()
{
    char buffer[500];
    sprintf(buffer,"online clients:\n");
    int i;
    for(i=0;i<5;i++)
    {
		if((client+i)->sockfd != clsockfd && (client+i)->sockfd !=0) // not its own details and valid clients only
		{
		    sprintf(buffer,"%sUniquename:%s Uniqueid:%d\n",buffer,(client+i)->uniquename,(client+i)->uniqueid);
		}
	}  
	write(clsockfd,buffer,sizeof(buffer));
}

//to send kill message to all the connected clients, just inserting into messageq and delivery process will takecare of it
void sendkillmessage()
{
    char buffer[200];
    int i;
    for(i=0;i<5;i++)
    {
		if((client+i)->sockfd == clsockfd) 
		{
		    sprintf(buffer,"client Uniquename:%s Uniqueid:%d is disconnected\n",(client+i)->uniquename,(client+i)->uniqueid);		    
		    break;
		}	
	}
	int j;        
    for(j=0;j<5;j++)
    {
		if((client+j)->sockfd != clsockfd && (client+j)->sockfd !=0) // not to itself and valid clients only
		{	
		    struct stmessage tmessage;
		    strcpy(tmessage.sendername, (client+i)->uniquename); // we got from last for loop where we got sender name using clsockfd
	        strcpy(tmessage.receivername,(client+j)->uniquename);
        	strncpy(tmessage.text,buffer,200);       
        	tmessage.sockfd=(client+j)->sockfd; // receiver sockfd 
		    insertmessage(tmessage);		    
		}
	}  		
}
//create shared memory for the client and message q
void createstructure()
{
	key_t key=0;
	key++;
	client=(struct stclient *)createshm(key,sizeof(struct stclient)*5);
	//printf("messageq size: %d\n",sizeof(struct stmessageq));
	key++;
	messageq=(struct stmessageq *)createshm(key,sizeof(struct stmessageq));
	messageq->head=0; //initialising head and tail
	messageq->tail=-1;
	key++;
	gbsemclient = createshm(key,sizeof(sem_t)*2);
	if(sem_init(gbsemclient,1,1))
	    perror("error in sem_init"); //initialising semaphore with 1 value and 1 for pshared value(for 	interprocess communication)
	key++;
	gbsemqueue = createshm(key,sizeof(sem_t)*2);
	if(sem_init(gbsemqueue,1,1))
	    perror("error in sem_init"); //initialising semaphore with 1 value and 1 for pshared value(for 		interprocess communication)

}
void* createshm(key_t key,int size)
{
	void *shm;
	if ((gshmid[gshmcount] = shmget(key, size, IPC_CREAT | 0666)) < 0) 
	{
		perror("shmget");
		exit(1);
	}
	//attaching the segment.
	if ((shm = shmat(gshmid[gshmcount], NULL, 0)) == (char *) -1) {
	perror("shmat");
	exit(1);
	}
	gshmcount++; // increase the overall counter of shared memory so that we can deallocate later on
	memset(shm,'\0',size);
	return shm;	
}

void exitprocess()
{ 	
    close(clsockfd);
	deattachshm();
	fclose(fp);       
	//printf("\ninside exitprocess");
	exit(1);
}
int deattachshm()
{
	if (shmdt(messageq) == -1) 
	{
        	perror("shmdt boiler");
        	exit(1);
    	}
   	if (shmdt(client) == -1) 
	{
        	perror("shmdt mixer");
        	exit(1);
    	}
 	/* Deallocate the shared memory segment.  */ 
	int i;
	for( i=0;i<gshmcount;i++)
		shmctl (gshmid[i], IPC_RMID, 0);
}
