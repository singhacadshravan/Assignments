#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/stat.h>
#include <dirent.h>  
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

int filltaskinfo(char *);
int filljobq(char *,int);
int readslave(FILE *);
int createjobq(FILE *,int);
int tasktype(int ,char *,int *);
int createtaskq(int );
int* createshm(key_t,int );
int deattachshm();
void exitprocess();
int memsetshm(int *,int ,int );
int settaskcounts(int,int ,int);
int printfinalcounts();

sem_t *gbsemaphore,*gmsemaphore,*gwsemaphore,*gfsemaphore; 
int *gboilerq,*gmixerq,*gwrapq,*gfreezeq;
int *gfboilerq,*gfmixerq,*gfwrapq,*gffreezeq;
int *ghtqpointer,*ghtfqpointer;// for allocating head tail for all queues in an array
int gshmid[100],gshmcount,gjbcount;

struct sttasktype{ int ittype;char cttype[100];int time;};
struct sttaskinfo { struct sttasktype ttype[50];} taskinfo[4];

struct stjobs{
 char jobname[100];
 int jobary[100];
 } jobs[500];

int slaveno[4];
struct taskcount{int task;int pid;int ttypecount[20];} tkcount[50];  // for each process id how much it performed
int tkcounter;

int main(int argc, char *argv[])
{
	filltaskinfo(argv[1]);
	filljobq(argv[2],atoi(argv[3]));	
	createtaskq(atoi(argv[3]));
	createslaves();
	scheduler(atoi(argv[3]));
	deattachshm();
	printfinalcounts();
	if(killpg(getpid(),SIGINT))perror("error killpg:");
}

/**
 * one by one go through the finished job queue and check which ones are finished and allocate next task of that job
 * 
 * @param totaljobs: input total number of jobs to be done * 
 * @return no of commands 
 */

int scheduler(int totaljobs)
{
	int task=0;
	int *tempq,jobcount=0;
	while(1)
	{
		sem_t *tsemaphore = (task==0?gbsemaphore:task==1?gmsemaphore:task==2?gwsemaphore:task==3?gfsemaphore:-1);
		tempq = (task==0?gfboilerq:task==1?gfmixerq:task==2?gfwrapq:task==3?gffreezeq:-1); // allocating correct queue
		
		if(ghtfqpointer[task*2]<=ghtfqpointer[task*2+1] && ghtfqpointer[task*2]!=-1) //if head<=tail of that queue then fetch the job ; first is tail
		{
			int jobid=(ghtfqpointer[task*2]==-1?-1111:tempq[3*ghtfqpointer[task*2]]);
			int tempt=tempq[3*ghtfqpointer[task*2]+1]; //tasktype 
			
			for(int i=0;i<97;i+=3) // loop through all the tasks in a job to see which one is completed
			{				
				if(jobs[jobid].jobary[i]==task && jobs[jobid].jobary[i+1]==tempt) // if same task and tasktype 
				{
					// if three consecutive 0's then nothing left in job
					if(jobs[jobid].jobary[i+3]==0 && jobs[jobid].jobary[i+4]==0 && jobs[jobid].jobary[i+5]==0)
					{	
						settaskcounts(task,tempq[3*ghtfqpointer[task*2]+2]	,tempq[3*ghtfqpointer[task*2]+1]);	
						printf("\n%s %s %s %d 0 Finished",jobs[jobid].jobname,task==0?"Boil":task==1?"Mix":task==2?"Wrap":task==3?
						"freeze":"Incorrect task", taskinfo[task].ttype[tempq[3*ghtfqpointer[task*2]+1]].cttype,tempq[3*ghtfqpointer[task*2]+2]);
						fflush(stdout);
						jobcount++;
						if(jobcount==totaljobs)return 1;// last job done
					}
					else// when not finished the whole job then allocate the next task to respective q
					{
		sem_wait(tsemaphore);
							 jobid=(ghtfqpointer[task*2]==-1?-1111:tempq[3*ghtfqpointer[task*2]]);
							 tempt=tempq[3*ghtfqpointer[task*2]+1]; //tasktype 
		
							int ntask=jobs[jobid].jobary[i+3]; //fetch the new task
							int *ntempq1=(ntask==0?gboilerq:ntask==1?gmixerq:ntask==2?gwrapq:ntask==3?gfreezeq:-1);
							// tempq is assigned the address of queue; 
							ntempq1[2*(ghtqpointer[ntask*2+1]+1)]=jobid; // job id in the tail section of q; +1 in tail as it will be new tail
							ntempq1[2*(ghtqpointer[ntask*2+1]+1)+1]=jobs[jobid].jobary[i+4];	// tasktype	
settaskcounts(task,tempq[3*ghtfqpointer[task*2]+2]	,tempt);								
							printf("\n%s %s %s processid %d %d waiting ",jobs[jobid].jobname,task==0?"Boil":task==1?"Mix":task==2?
							"Wrap":task==3?"freeze":"Incorrect task",taskinfo[task].ttype[tempq[3*ghtfqpointer[task*2]+1]].cttype,
							tempq[3*ghtfqpointer[task*2]+2],waitingtask(i,jobid));
							fflush(stdout);
							if(ghtqpointer[2*ntask]==-1)ghtqpointer[2*ntask]=0;// set qu for new taskt
							++ghtqpointer[2*ntask+1];//tail is increased for the new taskt
		sem_post(tsemaphore);					
					}
					ghtfqpointer[task*2]++;// increased the head of finished q
					break;
				}	
			}
		}
		task++;
		if(task==4)task=0;// to loop through all the task again
	}
}
/**
 * how many task are waiting to be done for this job
 * 
 * @param start: what is currently being done
  *@param jobid: which jobid we are working on
 * @return no of task still be done
 */

int waitingtask(int start,int jobid)
{
	int counter=0;
	for(int i=start;i<97;i+=3) // loop through all the tasks in a job to see which ones are incomplete
	{
		// if three consecutive 0's then nothing left in job
		if(jobs[jobid].jobary[i+3]==0 && jobs[jobid].jobary[i+4]==0 && jobs[jobid].jobary[i+5]==0)
		return counter;
	counter++;
	}
}
/**
 * each slave process will run this function to process the task * 
 * @param task: what task to be done by slave 
 * @return thestatus of slave process
 */

int slaveprocess(int task)
{
	signal(SIGINT, exitprocess);	
	int *tempq = (task==0?gboilerq:task==1?gmixerq:task==2?gwrapq:task==3?gfreezeq:-1);
	sem_t *tsemaphore = (task==0?gbsemaphore:task==1?gmsemaphore:task==2?gwsemaphore:task==3?gfsemaphore:-1);
	//printf("task:%d\n",task);
	while(1)
	{
		int jobid,temptype,status=0;
		
		sem_wait(tsemaphore);
		if(ghtqpointer[task*2]<=ghtqpointer[task*2+1] && ghtqpointer[task*2]!=-1) 		
		{
			jobid=(ghtqpointer[task*2] == -1 ?-1:tempq[2*ghtqpointer[task*2]]);
			temptype=(ghtqpointer[task*2] == -1?-1:tempq[2*ghtqpointer[task*2]+1]);
		
			++ghtqpointer[task*2]; // this needs to be put inside semaphore so that no other process can run this job;head is increased			
			// once finished, update the finished queues
			int *tempq1 = (task==0?gfboilerq:task==1?gfmixerq:task==2?gfwrapq:task==3?gffreezeq:-1);
			if(ghtfqpointer[task*2]==-1)ghtfqpointer[task*2]=0; // make the head 0	do these two line in semaphores		
			ghtfqpointer[task*2+1]++;
			tempq1[3*(ghtfqpointer[task*2+1])]=jobid;//
			tempq1[3*(ghtfqpointer[task*2+1])+1]=temptype;
			tempq1[3*(ghtfqpointer[task*2+1])+2]=getpid();//store the processid who has the work			
			status=1;			
			//printf("\nslaveprocess task:%d jobid:%d tasktype:%d processid:%d finishhead:%d finishtail:%d qhead:%d qtail:%d",
			//task,jobid,temptype,getpid(),ghtfqpointer[task*2],ghtfqpointer[task*2+1],ghtqpointer[task*2],ghtqpointer[task*2+1]);
			fflush(stdout);
		}
			sem_post(tsemaphore);
			if(status==1)usleep(taskinfo[task].ttype[temptype].time * 1000); // processing the job		
	}
}
/**
 * Task queues,finished task queues, head-tail queue are created here 
 * @param totaljobs: number of jobs to be done in total
 * @return thestatus of this process
 */

int createtaskq(int totaljobs)
{
	key_t key=0;	

	int qsize= sizeof(totaljobs)*sizeof(int)*2+4;
	gboilerq=createshm(key,qsize);//0
	key++;
	gmixerq=createshm(key,qsize);//1
	key++;
	gwrapq=createshm(key,qsize);
	key++;
	gfreezeq=createshm(key,qsize);
	key++;
	ghtqpointer=createshm(key,sizeof(int)*2*4+2);// for allocating head tail for all queues in an array
	
	key++;	
	gfboilerq=createshm(key,qsize*2);// 4
	key++;
	gfmixerq=createshm(key,qsize*2);//5
	key++;
	gfwrapq=createshm(key,qsize*2);
	key++;
	gffreezeq=createshm(key,qsize*2);	
	key++;
	ghtfqpointer=createshm(key,sizeof(int)*2*4+2);// for allocating head tail for all finished queues in an array
	key++;
	gbsemaphore = createshm(key,sizeof(sem_t)*2);
	if(sem_init(gbsemaphore,1,1))perror("error in sem_init"); //initialising semaphore with 1 value and 1 for pshared value(for interprocess communication)
	key++;
	gmsemaphore = createshm(key,sizeof(sem_t)*2);
	if(sem_init(gmsemaphore,1,1))perror("error in sem_init"); //initialising semaphore with 1 value and 1 for pshared value(for interprocess communication)key++;
	gwsemaphore = createshm(key,sizeof(sem_t)*2);
	if(sem_init(gwsemaphore,1,1))perror("error in sem_init"); //initialising semaphore with 1 value and 1 for pshared value(for interprocess communication)key++;
	gfsemaphore = createshm(key,sizeof(sem_t)*2);
	if(sem_init(gfsemaphore,1,1))perror("error in sem_init"); //initialising semaphore with 1 value and 1 for pshared value(for interprocess communication)
	
	memset(gfboilerq,'\0',qsize*2);
	memset(gboilerq,'\0',qsize);
	
	memsetshm(ghtqpointer,8,-1);
	memsetshm(ghtfqpointer,8,-1);
	
	// fill the jobs into the q
	int *tempq;
	int bq=0,mq=0,wq=0,fq=0; // individual markers if any job doesnt start from boiler
	int i=0;

	for(i=0;i< totaljobs;i++)
	{
		tempq=jobs[i].jobary[0]==0?gboilerq:jobs[i].jobary[0]==1?gmixerq:jobs[i].jobary[0]==2?gwrapq:jobs[i].jobary[0]==3?gfreezeq:-1;
		// tempq is assigned the address of queue; currently only boiler is assigned initially not others
		tempq[2*i]=i; // job id
		tempq[2*i+1]=jobs[i].jobary[1];	// tasktype

		if(ghtqpointer[0]==-1){ghtqpointer[0]=0;}
		ghtqpointer[1]++;// setting the tail of the queue
		//printf("createtaskq: job id:%d tasktype:%d totaljobs:%d qhead:%d qtail:%d\n",tempq[2*i],tempq[2*i+1],totaljobs,ghtqpointer[0],ghtqpointer[1]);	
		fflush(stdout);		
	}
	//ghtqpointer[0]=-1; // only set for boiler queue; first is head and second is tail	
}
int memsetshm(int *pointer,int size,int set)
{
for(int i=0;i<size;i++)
*(++pointer)=set;
}

int createslaves()
{
	int pid,task=0;
	while(task<4)
	{
		for(int i=0;i<slaveno[task];i++)
		{			
			if((pid=fork())==0)
			{ 
				//usleep(100);
				slaveprocess(task);		
			}
			else if (pid  < 0)perror("Forking child process failed:");	
			else
			{
				printf("\n%s processid:%d",task==0?"Boil":task==1?"Mix":task==2?"Wrap":task==3?"freeze":"Incorrect task",pid);
				fflush(stdout);
			}
		}
	task++;
	}	
}
void exitprocess()
{ 	
	deattachshm();
	//printf("\ninside exitprocess");
	exit(1);
}

int* createshm(key_t key,int size)
{
	int *shm;
	//printf("size:%d",size);
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
	//printf(" %d\n",shm);
	return shm;	
}

int filljobq(char *filename,int totaljobs)
{
	FILE *fpread;	
	fpread=(FILE *)fopen(filename, "r"); //reading slave.info
	if(fpread==NULL)
	{		
		perror("fopen():");
		return -1; // return when error 	
	}	
	createjobq(fpread,totaljobs);	
	fclose(fpread);
}
/**
 * prints the final counts of each process
 * 
 *  * @return no of commands 
 */

int printfinalcounts()
{
	int c=0;
	char *str1=(char *)calloc(sizeof(char),300);
	while(c<4)
	{
		for(int i=0;i<tkcounter;i++)
		{
			if(tkcount[i].task==c )
				{
					int count=0;
					for(int j=0;j<20 && strcmp(taskinfo[c].ttype[j].cttype,"")!=0 ;j++)
						{
							count+=tkcount[i].ttypecount[j];
							//printf("\n%s %d",taskinfo[c].ttype[j].cttype,tkcount[i].ttypecount[j]);
							sprintf(str1,"%s %s %d",str1,taskinfo[c].ttype[j].cttype,tkcount[i].ttypecount[j]);
							//strcat(str1,tkcount[i].ttypecount[j]);
						}
					printf("\n%d %d %s %s",tkcount[i].pid,count,c==0?"Boil":c==1?"Mix":c==2?"Wrap":c==3?"freeze":"Incorrect task",str1);					
					fflush(stdout);
					memset(str1,'\0',300);	
				}
		}
		c++;
	}
}
int settaskcounts(int task,int pid,int ttype)
{
	int i=0;
	for( i=0;i<tkcounter;i++)
	{
		if(tkcount[i].task==task && tkcount[i].pid==pid)
			{tkcount[i].ttypecount[ttype]++; return 1;}
	}
	if(tkcounter==i){
	tkcount[i].task=task;
	tkcount[i].pid=pid;
	tkcount[i].ttypecount[ttype]++;
	tkcounter++;
	}
}

int filltaskinfo(char *filename)
{
	FILE *fpread;	
	fpread=(FILE *)fopen(filename, "r"); //reading slave.info
	
	if(fpread==NULL)
	{		
		perror("fopen():");
		return -1; // return when error 
	}
	readslave(fpread);
	fclose(fpread);	
}

int createjobq(FILE *fpread,int totaljobs)
{
	char *line=(char *)calloc(sizeof(char),500); // text will be read from file into this buffer
	int count=0;

	while(fgets(line,500,fpread)!=NULL) // buf is assigned a row from file
	{
		char *token = strtok(line, " ");
		int argcount=0;
		
		strcpy(jobs[count].jobname,token);
		//printf( "%s ",jobs[count].jobname);
		token = strtok(NULL, ":");
		int time=0;
		while(token != NULL )
		{
			//first value is task, second is tasktype third is minutes that tasktype takes
			jobs[count].jobary[argcount]=strcmp(token,"boil")==0?0:strcmp(token,"mix")==0?1:strcmp(token,"wrap")==0?2:strcmp(token,"freeze")==0?3:-1;		
			
			token = strtok(NULL, " ");
			if(*(token+strlen(token)-1)=='\n')
			{
				char *temp=(char *)calloc(sizeof(char),30); // text will be read from file into this buffer
				token=strncpy(temp,token,strlen(token)-1);
				//printf("newline %s",token);
			}
			int temp=tasktype(jobs[count].jobary[argcount],token,&time);
			++argcount;
			jobs[count].jobary[argcount]=temp;// tasktype
			++argcount;
			jobs[count].jobary[argcount]=time;
				
			//printf( "task:%d tasktype:%d time:%d ", jobs[count].jobary[argcount-2],jobs[count].jobary[argcount-1],jobs[count].jobary[argcount] );
			argcount++;
			token = strtok(NULL, ":"); // for next task type		
		}
		//printf("\n");
		count++;
	}
	gjbcount=count;
	free(line);
	// for filling rest of the job queue based on existing data 
	int j=0;
	for(int i=count;i<totaljobs;i++)
	{
		strcpy(jobs[i].jobname,jobs[j].jobname);
		for(int x=0;x<97;x+=3)
		{	
			jobs[i].jobary[x]=jobs[j].jobary[x];
			jobs[i].jobary[x+1]=jobs[j].jobary[x+1];
			jobs[i].jobary[x+2]=jobs[j].jobary[x+2];
		}
		j++;
	if(j==count)j=0;	
	}
}

int tasktype(int task,char ttype[100],int *time)
{
	for(int i=0;i<=50,strlen(taskinfo[task].ttype[i].cttype)!=0;i++)
	if(strcmp(taskinfo[task].ttype[i].cttype,ttype)==0)
	{
		*time=taskinfo[task].ttype[i].time ; 
		return i;
	}
	return -1;
}

int readslave(FILE *fpread)
{
	char *line=(char *)calloc(sizeof(char),500); // text will be read from file into this buffer
	int task=0;
	
	while(fgets(line,500,fpread)!=NULL) // buf is assigned a row from file
	{	
		char *token = strtok(line, " ");		
		int argcount=0;
		task=strcmp(token,"boil")==0?0:strcmp(token,"mix")==0?1:strcmp(token,"wrap")==0?2:strcmp(token,"freeze")==0?3:-1;		 
		token = strtok(NULL, " ");		
		slaveno[task]=atoi(token);
		//printf( "%d ", slaveno[task] );
		token = strtok(NULL, " ");
		
		while( token != NULL )
		{
			
			taskinfo[task].ttype[argcount].ittype=argcount;		
			strcpy(taskinfo[task].ttype[argcount].cttype,token);
			token = strtok(NULL, " ");				
			taskinfo[task].ttype[argcount].time=atoi(token);	

			//printf( "%d %s %d ", taskinfo[task].ttype[argcount].ittype,taskinfo[task].ttype[argcount].cttype,taskinfo[task].ttype[argcount].time );
			argcount++;
			token = strtok(NULL, " "); // for next task type		
		}
		//count++;
		//printf( "\n");
	}
	free(line);
}
int deattachshm()
{
	if (shmdt(gboilerq) == -1) 
	{
        	perror("shmdt boiler");
        	exit(1);
    	}
    	if (shmdt(gmixerq) == -1) 
	{
        	perror("shmdt mixer");
        	exit(1);
    	}
    	if (shmdt(gwrapq) == -1) 
	{
        	perror("shmdt wrap");
        	exit(1);
    	}
    	if (shmdt(gfreezeq) == -1) 
	{
        	perror("shmdt freeze");
        	exit(1);
    	}
    	if (shmdt(ghtqpointer) == -1) 
	{
        	perror("shmdt ghtqpointer");
        	exit(1);
    	}
    	/* Deallocate the shared memory segment.  */ 
	for(int i=0;i<gshmcount;i++)
		shmctl (gshmid[i], IPC_RMID, 0);
		
}
