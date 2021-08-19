<template>
<div>
    <b-modal id="helpinfo-modal" title="Help" size="lg" ok-only>
        <HelpInfo contentId="sessions" :fullPage="true">
        </HelpInfo>
    </b-modal>
    <div class="ml-2 text-right">
        <b-link v-b-modal="'helpinfo-modal'" style="color:white">
            <i class="fa fa-question-circle"></i>
        </b-link>
    </div>
    <b-form-checkbox class="ml-2" switch v-model="archived">Archived Sessions Only</b-form-checkbox>

    <div class="mt-4"></div>

    <div v-if="$apollo.queries.sessions.loading">Loading..</div>

    <div v-else>

        <b-card-text v-if="rows > 0">
            <b-button :disabled="selected.length <= 1" v-on:click="runCompare" variant="info" size="sm">
                <span v-if="selected.length > 1">Compare: {{ selected.length }}</span>
                <span v-else>Compare: select at least 2</span>

            </b-button>
            <b-form-checkbox-group v-model="selected">
                <b-table id="datatable" hover table-busy :items="sessionList" :fields="fields" :dark="false" :small="false" :bordered="false" :outlined="false" :borderless="false">
                    <template v-slot:cell(selector)="data">
                        <b-form-checkbox :value="data.item.ident"></b-form-checkbox>
                    </template>

                    <template v-slot:cell(link)="data">
                        <!-- `data.value` is the value after formatted by the Formatter -->
                        <router-link :to="`/session/view/${data.item.ident}`">{{data.item.ident }}</router-link>
                    </template>

                    <template v-slot:cell(taskId)="data">
                        <!-- `data.value` is the value after formatted by the Formatter -->
                        <router-link v-if="data.item.task" :to="`/task/view/${data.item.task.ident}`">{{data.item.task.ident }}</router-link>
                    </template>
                    <template v-slot:cell(envSpecId)="data">
                        <!-- `data.value` is the value after formatted by the Formatter -->
                        <router-link :to="`/env_spec/view/${data.item.envSpecId}`">{{ data.item.envSpecId }}</router-link>
                    </template>

                </b-table>
                <div>
                    <b-button-toolbar key-nav>
                        <b-button-group class="mx-1">
                            <b-button @click="previousPage()">&lsaquo;</b-button>
                        </b-button-group>
                        <b-button-group class="mx-1">
                            <b-button @click="nextPage()">&rsaquo;</b-button>
                        </b-button-group>
                    </b-button-toolbar>
                </div>
            </b-form-checkbox-group>
            <p>{{ error }}</p>
        </b-card-text>
        <b-card-text v-else>No Sessions Found</b-card-text>
    </div>

</div>
</template>

<script>
// @ is an alias to /src
import axios from "axios";
import {
    appConfig
} from "../app.config";
import {
    formatNum,
    timedelta,
    timepretty
} from "../funcs";
import gql from "graphql-tag";

const fields = [{
        key: "selector",
        label: "",
    },

    {
        key: "link",
        label: "Session Id",
        sortable: true,
    },
    {
        key: "sessionType",
        label: "Session Type",
        sortable: true,
    },
    {
        key: "envSpecId",
        label: "Environment",
    },
    {
        key: "agentId",
        label: "Agent",
    },
    {
        key: "created",
        label: "Created",
        sortable: true,
        // formatter: timepretty,
    },
    {
        key: "status",
        label: "State",
    },

    {
        key: "taskId",
        label: "Task",
    }
];
const GET_ALL_SESSIONS = gql `
  query GetSessions($first: Int, $after: String, $before:String, $archived:Boolean) {
    sessions: allSessions(
      sort: CREATED_DESC
      first: $first
      before: $before
      after: $after,
      filters: {
            archived:$archived  
        }
    ) {
      pageInfo {
        startCursor
        endCursor
        hasNextPage
        hasPreviousPage
      }
      edges {
        node {
            id
          ident
          created
          envSpecId
          label
          comments
          status
          config
          sessionType
          agentId
            task {
                id
            ident
        }
        
          summary 
        }
      }
    }
  }
`;
export default {
    name: "Sessions",
    components: {
        // SessionList
    },
    apollo: {
        sessions: {
            query: GET_ALL_SESSIONS,
            variables() {
                return {
                    first: this.first,
                    last: this.last,
                    before: this.startCursor,
                    after: this.endCursor,
                    archived: this.archived
                };
            },
        },
    },
    data() {
        return {
            sessions: [],
            first: 10000000, //TODO: Fix pagination bug
            startCursor: "",
            endCursor: "",
            searchQuery: "",
            fields: fields,
            error: "",
            selected: [],
            archived: false
        };
    },
    methods: {
        runCompare() {
            this.$router.push({
                path: `/session/compare?uids=` + this.selected.join(","),
            });
        },
        previousPage() {
            this.startCursor = this.sessions.pageInfo.endCursor;
            this.endCursor = "";
            this.$apollo.queries.sessions.refetch();
        },
        nextPage() {
            this.startCursor = "";
            this.endCursor = this.sessions.pageInfo.startCursor;
            this.$apollo.queries.sessions.refetch();
        },
    },

    computed: {
        sessionList() {
            if (!this.sessions.edges) return [];

            return this.sessions.edges.map((session) => session.node);
        },
        rows() {
            if (!this.sessions.edges) return 0;
            return this.sessions.edges.length;
        },
    },
    // Fetches posts when the component is created.
    created() {
        //asdf
    },
};
</script>

<style>
/* a.page-link{
   color: black;
 } */
</style>
