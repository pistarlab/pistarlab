<template>
<div>

    <h1><i class="fa fa-sitemap"></i> Components</h1>

    <div class="mt-4"></div>
    <b-alert show variant="warning">WARNING: Components are under development.</b-alert>

    <p>Reusable pieces of functionality that can be used in Agents or other Components.</p>
    <div class="mt-2"></div>
    <b-container fluid>

        <b-row>
            <b-col>

                <b-card>
                    <div v-if="$apollo.queries.componentSpecs.loading">Loading..</div>
                    <div v-else>
                        <b-card-text v-if="rows > 0">
                            <b-form-checkbox-group v-model="selected">
                                <b-table id="datatable" hover table-busy :items="componentSpecList" :fields="fields" :dark="false" :small="false" :bordered="false" :outlined="false" :borderless="false">

                                    <template v-slot:cell(link)="data">
                                        <!-- `data.value` is the value after formatted by the Formatter -->
                                        <router-link :to="`/component_spec/view/${data.item.ident}`">{{data.item.ident }}</router-link>
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
                        <b-card-text v-else>No Component Specs Found</b-card-text>
                    </div>
                </b-card>
            </b-col>
        </b-row>
    </b-container>
</div>
</template>

<script>
import gql from "graphql-tag";

const fields = [

    {
        key: "link",
        label: "Spec Id",
        sortable: true,
    },
    {
        key: "category",
        label: "Type",
        sortable: true,
    },
    {
        key: "extensionId",
        label: "extension",
        sortable: true,
    },
    {
        key: "created",
        label: "Created",
        sortable: true,
        // formatter: timepretty,
    },

];
const GET_ALL_COMPONENTS = gql `
  query GetComponentSpecs($first: Int, $after: String, $before:String) {
    componentSpecs: allComponentSpecs(
      sort: CREATED_DESC
      first: $first
      before: $before
      after: $after
    ) {
      pageInfo {
        startCursor
        endCursor
        hasNextPage
        hasPreviousPage
      }
      edges {
        node {
          ident
          created
          category
            extensionId
          config

        }
      }
    }
  }
`;
export default {
    name: "ComponentSpecs",
    components: {
        // ComponentSpecList
    },
    apollo: {
        // Simple query that will update the 'hello' vue property
        componentSpecs: {
            query: GET_ALL_COMPONENTS,
            variables() {
                return {
                    first: this.first,
                    last: this.last,
                    before: this.startCursor,
                    after: this.endCursor,
                };
            },
        },
    },
    data() {
        return {
            componentSpecs: [],
            first: 20,
            startCursor: "",
            endCursor: "",
            searchQuery: "",
            fields: fields,
            error: "",
            selected: [],
        };
    },
    methods: {
        previousPage() {
            this.startCursor = this.componentSpecs.pageInfo.endCursor;
            this.endCursor = "";
            this.$apollo.queries.componentSpecs.refetch();
        },
        nextPage() {
            this.startCursor = "";
            this.endCursor = this.componentSpecs.pageInfo.startCursor;
            this.$apollo.queries.componentSpecs.refetch();
        },
    },

    computed: {
        componentSpecList() {
            if (!this.componentSpecs.edges) return [];

            return this.componentSpecs.edges.map((componentSpec) => componentSpec.node);
        },
        rows() {
            if (!this.componentSpecs.edges) return 0;
            return this.componentSpecs.edges.length;
        },
    },
    // Fetches posts when the component is created.
    created() {
        // 
    },
};
</script>

<style>
/* a.page-link{
   color: black;
 } */
</style>
