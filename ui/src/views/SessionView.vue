<template>
<div v-if="session && session.sessionType">
    <SessionViewer v-if="session.sessionType == 'RL_SINGLEPLAYER_SESS'" @update="refreshData(true)" :session="session" :uid="uid"></SessionViewer>
    <MultiSessionViewer v-else-if="session.sessionType.startsWith('RL_MULTIPLAYER_')" :session="session" :uid="uid"  @update="refreshData(true)" ></MultiSessionViewer>
    <GenericSessionViewer v-else :session="session" :uid="uid"  @update="refreshData(true)" ></GenericSessionViewer>
</div>
<div v-else>
    Loading...
</div>
</template>

<script>
import gql from "graphql-tag";

import SessionViewer from "../components/SessionViewer.vue";
import MultiSessionViewer from "../components/MultiSessionViewer.vue";
import GenericSessionViewer from "../components/GenericSessionViewer.vue";


const GET_SESSION = gql `
 query GetSession($ident: String) {
  session(ident: $ident) {
    id
    ident
    envSpecId
    envSpec {
        id
        displayedName
      environment {
        id
        ident
        displayedName
      }
    }
    sessionType
    created
    config
    runInfo
    status
    archived
    task {
      id
      ident
      status
    }
    createdTimestamp
    parentSessionId
    parentSession {
      id
      ident
      sessionType
      task {
          id
        ident
      }
    }
    agentId
    agent {
      id
      ident
      name
      status
      created
      config
      specId
      spec {
        id
        displayedName
        ident
        config
        meta
      }
    }
    childSessions {
      edges {
        node {
          id
          ident
          envSpecId
          created
          config
          runInfo
          sessionType
          status
          task {
            id
            ident
          }
          agent {
            id
            ident
            specId
            name
    
          }
          createdTimestamp
          summary 
        }
      }
    }
    summary 
    
  }
}
`;

export default {
    name: "Session",
    components: {
        SessionViewer,
        MultiSessionViewer,
        GenericSessionViewer
    },
    apollo: {
        // Simple query that will update the 'hello' vue property
        session: {
            query: GET_SESSION,
            variables() {
                return {
                    ident: this.uid
                };
            },
        },
    },
    data() {
        return {
            session: {},
        };
    },
    props: {
        uid: String,
    },
    computed: {

    },
    mounted() {
        //
    },
    methods: {
        refreshData(force=false) {
            if (force || !this.session || this.session.status == null || (this.session.status && (this.session.status == "RUNNING" || this.session.status == "CREATED"))) {
                this.$apollo.queries.session.refetch();
            } 
            // else {
            //     clearInterval(this.timer);
            //     return
            // }
        }
    },
    created() {
        console.log(this.uid);
        this.timer = setInterval(this.refreshData, 3000);
    },
    beforeDestroy() {
        if (this.es) {
            this.es.close();
        }
        clearInterval(this.timer);
    },
};
</script>

<style>

</style>
