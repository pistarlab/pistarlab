<template>
<div>
    <div class="mt-2"></div>
    <b-form-input v-model="searchtext" placeholder="Search" style="width:250px;" class='ml-auto'></b-form-input>
    <div class="mt-1"></div>

    <b-pagination v-model="currentPage" :total-rows="totalCount" :per-page="pageSize"></b-pagination>

    <b-container fluid>
        <div v-for="(spec,idx) in envSpecListPage" v-bind:key="idx">
            <b-row>
                <b-col cols=3 class="text-center">
                    <div class="mt-2">
                        <router-link :to="`/env_spec/view/${spec.ident}`"> {{ spec.displayedName }}</router-link>
                    </div>
                    <b-card-img class="mt-1" :src="`${appConfig.API_URL}/api/env_preview_image/${spec.ident}`" alt="" style="width:100px"></b-card-img>

                </b-col>
                <b-col>
                    <div class="mt-3"></div>
                    <div>
                        <span class="data_label mt-1">Environment: </span>
                        <span>{{spec.environment.ident}}</span>
                    </div>
                    <div>
                        <span class="data_label mt-1">Version: </span>
                        <span>{{spec.environment.version}}</span>
                    </div>
                    <div>
                        <span class="data_label mt-1">Extension: </span>
                        <span>{{spec.environment.extensionId}}: {{spec.environment.extensionVersion}}</span>
                    </div>
                    <div>
                        <span class="data_label mt-1">Type: </span>
                        <span>{{spec.envType}}</span>
                    </div>
                    <div>
                        <span class="data_label mt-1">Description: </span>
                        <span>{{spec.description}}</span>
                    </div>
                </b-col>
                <b-col cols=2>
                    <div class="mt-3"></div>
                    <div>
                        <b-button size="sm" variant="primary" @click="select(spec.ident)">Select</b-button>
                    </div>

                </b-col>
            </b-row>

            <hr />
        </div>
    </b-container>
    <b-pagination v-model="currentPage" :total-rows="totalCount" :per-page="pageSize"></b-pagination>
    <div class="mt-2"></div>
</div>
</template>

<script>
import axios from "axios";
import {
    appConfig
} from "../app.config";
import {
    timedelta,
    timepretty
} from "../funcs";
import gql from "graphql-tag";

const GET_ENV_SPECS = gql `
  query {
    envSpecs {
      id
      ident
      displayedName
      config
      meta
      
      envType
      description
      environment{
          id
          ident
          extensionId
            version
            extensionVersion
            disabled
      }
    }
  }
`;

export default {
    props: {
        //
    },
    apollo: {
        envSpecs: GET_ENV_SPECS,

    },
    data() {
        return {
            envSpecs: [],
            selectedExistingAgent: null,
            searchtext: "",
            pageSize: 12,
            currentPage: 1

        };
    },
    mounted() {
        //
    },
    methods: {
        select(uid) {
            this.$emit('click', uid)

        },

    },
    computed: {

        envSpecList() {
            if (this.envSpecs.length == 0) return [];
            else {
                if (this.searchtext != "") {
                    return this.envSpecs.filter((v) =>
                        !v.environment.disabled && v.displayedName.toLowerCase().includes(this.searchtext.toLowerCase())
                    )
                } else {
                    return this.envSpecs.filter((v) => !v.environment.disabled)

                }
            }
        },
        totalCount() {
            return this.envSpecList.length
        },
        envSpecListPage() {
            return this.envSpecList.slice(parseInt((this.currentPage -1) * this.pageSize), parseInt(this.currentPage  * this.pageSize))

        },

    },
    // Fetches posts when the component is created.
    created() {
        //

    },
    beforeDestroy() {
        //

    }
};
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->

<style scoped lang="scss">

</style>
