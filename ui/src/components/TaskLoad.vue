<template>
<div>
    <div>
        Task ID: <router-link :to="`/task/view/${uid}`">{{uid}}</router-link>
    </div>
    <div v-if="task.status == 'RUNNING'">
        Loading...
    </div>
    <div v-else>
        Status: {{task.status}}
    </div>
    <div>
        <LogViewer :nocard="true" :logStreamUrl="`${appConfig.API_URL}/api/stream/entity_logs/task/${uid}`"> </LogViewer>
    </div>
</div>
</template>

<script>
import gql from "graphql-tag";
import {
    appConfig
} from "../app.config";
import LogViewer from "./LogViewer.vue";

const GET_TASK = gql `
  query GetTask($ident: String!) {
    task(ident: $ident) {
      id
      ident
      status
      statusMsg
      config
      created
      summary
      primarySession{
          id
          ident
      }
     
    }
  }
`;

export default {
    name: "TaskLoad",
    components: {
        LogViewer
    },
    apollo: {
        task: {
            query: GET_TASK,
            variables() {
                return {
                    ident: this.uid,
                };
            },
            pollInterval: 2000,
        },
    },
    data() {
        return {
            task: {},
            error: "",
            appConfig,
            intervalTimer: null,
        };
    },
    props: {
        uid: String,
        redirectToSession: Boolean
    },
    created() {
        console.log(this.uid);
        this.intervalTimer = setInterval(() => {
            console.log("Checking")
            if (!this.$apollo.queries.task.loading) {
                if (this.task.status && this.task.status == "RUNNING") {

                    if (this.task.primarySession && this.task.primarySession.ident) {
                        this.$router.push({
                            path: '/session/view/' + this.task.primarySession.ident
                        })
                    } else {
                        console.log("Refetch")
                        this.$apollo.queries.task.refetch();
                    }

                }
                else if(this.task.status && (this.task.status != "RUNNING" && this.task.status != "CREATED")){
                      this.$router.push({
                            path: '/task/view/' + this.task.ident
                        })
                }
            }
        }, 1000);
    },
    beforeDestroy() {
        if (this.intervalTimer) clearInterval(this.intervalTimer);
    },
};
</script>

<style>

</style>
